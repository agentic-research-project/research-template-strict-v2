import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA

class GaussianPyramid(nn.Module):
    def __init__(self, sigma=2.0, levels=2):
        super().__init__()
        self.sigma = sigma
        self.levels = levels

    def _gaussian_blur(self, x):
        kernel_size = int(6 * self.sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel_1d = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        kernel_2d = kernel_2d.repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel_2d, padding=radius, groups=x.shape[1])

    def forward(self, x):
        # x: B, C, H, W
        lf = self._gaussian_blur(x)
        hf = x - lf
        return lf, hf

class EfficientViTEncoder(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        # Use ResNet18 as lightweight alternative to EfficientViT-M4
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove avgpool and fc
        
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Extract multi-scale features
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [5, 6, 7]:  # layer2, layer3, layer4 outputs
                features.append(x)
        return features

class HFNoiseGate(nn.Module):
    """Channel-wise learnable noise gate that handles multi-scale ResNet18 features.

    ResNet18 multi-scale outputs (layer2=128, layer3=256, layer4=512).
    Gates are pre-registered for all three scales so the optimizer never sees
    an empty parameter list.
    """
    RESNET18_SCALES = [128, 256, 512]

    def __init__(self, channels=512):
        super().__init__()
        self.ref_channels = channels
        # Pre-register gates for all expected channel sizes
        self._gates = nn.ModuleDict({
            str(c): nn.Sequential(nn.Conv1d(c, c, 1), nn.Sigmoid())
            for c in self.RESNET18_SCALES
        })

    def _get_gate(self, c: int) -> nn.Module:
        key = str(c)
        if key not in self._gates:
            self._gates[key] = nn.Sequential(
                nn.Conv1d(c, c, 1),
                nn.Sigmoid()
            )
        return self._gates[key]

    def forward(self, hf_features):
        # Apply channel-wise gating per scale
        gated = []
        for feat in hf_features:
            B, C, H, W = feat.shape
            gate = self._get_gate(C).to(feat.device)
            feat_pooled = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)  # B, C
            gate_weights = gate(feat_pooled.unsqueeze(-1)).squeeze(-1)  # B, C
            gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1)  # B, C, 1, 1
            gated.append(feat * gate_weights)
        return gated

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, content, style_stats):
        # content: B, C, H, W
        # style_stats: (mean, std) from reference patches
        B, C, H, W = content.shape
        content_flat = content.view(B, C, -1)
        content_mean = content_flat.mean(dim=2, keepdim=True)
        content_std = content_flat.std(dim=2, keepdim=True) + self.eps
        
        normalized = (content_flat - content_mean) / content_std
        
        # Apply reference style statistics if available
        if style_stats is not None:
            style_mean, style_std = style_stats
            styled = normalized * style_std + style_mean
        else:
            styled = normalized
            
        return styled.view(B, C, H, W)

class SimpleFAISSReplacement:
    """Simple replacement for FAISS using cosine similarity"""
    def __init__(self, d):
        self.d = d
        self.index_data = None
        
    def add(self, vectors):
        if self.index_data is None:
            self.index_data = vectors.copy()
        else:
            self.index_data = np.vstack([self.index_data, vectors])
    
    def search(self, queries, k):
        if self.index_data is None:
            return np.array([]), np.array([])
            
        # Normalize vectors for cosine similarity
        queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        index_norm = self.index_data / np.linalg.norm(self.index_data, axis=1, keepdims=True)
        
        # Compute similarity scores
        similarities = np.dot(queries_norm, index_norm.T)
        
        # Get top k indices and scores
        top_k_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
        top_k_scores = np.take_along_axis(similarities, top_k_indices, axis=1)
        
        # Convert similarity to distance (FAISS-like format)
        distances = 1 - top_k_scores
        
        return distances, top_k_indices

class FreqConsensusAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Components
        self.freq_decomposer = GaussianPyramid(
            sigma=config.get('freq_sigma', 2.0),
            levels=config.get('freq_levels', 2)
        )
        self.encoder = EfficientViTEncoder(frozen=True)
        self.hf_gate = HFNoiseGate(channels=512)  # ResNet18 final channels
        self.ada_norm = AdaptiveInstanceNorm()
        
        # FAISS replacement
        self.reference_index = SimpleFAISSReplacement(d=512)
        self.k = config.get('k_neighbors', 5)
        
    def forward(self, x, reference_features=None):
        # Frequency decomposition
        lf, hf = self.freq_decomposer(x)
        
        # Extract features
        lf_features = self.encoder(lf)
        hf_features = self.encoder(hf)
        
        # Apply HF noise gate
        gated_hf = self.hf_gate(hf_features)
        
        # Use final layer features for anomaly detection
        if len(lf_features) > 0:
            lf_feat = lf_features[-1]
        else:
            lf_feat = lf
            
        if len(gated_hf) > 0:
            hf_feat = gated_hf[-1]
        else:
            hf_feat = hf
        
        # Global average pooling to get patch-level features
        lf_vec = F.adaptive_avg_pool2d(lf_feat, 1).squeeze(-1).squeeze(-1)
        hf_vec = F.adaptive_avg_pool2d(hf_feat, 1).squeeze(-1).squeeze(-1)
        
        # Combine features
        combined_features = lf_vec + hf_vec
        
        return combined_features

def build_model(config):
    """Build FreqConsensus-AD model"""
    return FreqConsensusAD(config)