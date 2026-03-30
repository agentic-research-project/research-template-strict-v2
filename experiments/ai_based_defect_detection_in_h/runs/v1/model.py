import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MultiCueCoarseGate(nn.Module):
    """Stage-1: Multi-Cue Coarse Gate with 4 parallel cues"""
    def __init__(self, acceptance_percentile=1.0):
        super().__init__()
        self.acceptance_percentile = acceptance_percentile
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.reference_stats = None
        
    def forward(self, patches):
        """patches: [B, 1, 256, 256]"""
        batch_size = patches.shape[0]
        device = patches.device
        
        # Sobel kernels are registered buffers and follow module device automatically
        
        # C1: Pixel variance
        variance = torch.var(patches.view(batch_size, -1), dim=1)
        
        # C2: Shannon entropy (8-bit histogram, GPU batchwise)
        patch_u8 = (patches[:, 0].clamp(0, 1) * 255).to(torch.long)
        hist = torch.zeros(batch_size, 256, device=device)
        hist.scatter_add_(1, patch_u8.view(batch_size, -1), torch.ones(batch_size, patch_u8[0].numel(), device=device))
        prob = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-8)
        entropy = -(prob * torch.log2(prob.clamp_min(1e-8))).sum(dim=1)
        
        # C3: Sobel-edge maximum
        sobel_x_resp = F.conv2d(patches, self.sobel_x, padding=1)
        sobel_y_resp = F.conv2d(patches, self.sobel_y, padding=1)
        edge_max = torch.max(torch.max(torch.abs(sobel_x_resp).view(batch_size, -1), dim=1)[0],
                            torch.max(torch.abs(sobel_y_resp).view(batch_size, -1), dim=1)[0])
        
        # C4: Placeholder for Mahalanobis (will be computed with encoder block-1)
        mahalanobis = torch.zeros(batch_size, device=device)
        
        cues = {
            'variance': variance,
            'entropy': entropy,
            'edge_max': edge_max,
            'mahalanobis': mahalanobis
        }
        
        if self.reference_stats is not None:
            # Apply acceptance bands: patch is trivially normal only if ALL cues are within-band
            mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            for cue_name, values in cues.items():
                if cue_name in self.reference_stats:
                    band = self.reference_stats[cue_name]
                    if isinstance(band, dict):
                        lower = band.get('lower', None)
                        upper = band.get('upper', None)
                        if lower is not None:
                            mask &= (values >= lower)
                        if upper is not None:
                            mask &= (values <= upper)
            return mask
        
        return cues

class FrozenEncoder(nn.Module):
    """Stage-2: Frozen shallow encoder with AdaIN style normalization"""
    def __init__(self, input_channels=3, adain_blocks=[1, 2]):
        super().__init__()
        self.adain_blocks = adain_blocks
        
        # Simple CNN blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def adain_normalize(self, x):
        """Robust AdaIN using median/MAD statistics"""
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        
        # Use median and MAD for robust statistics
        median = torch.median(x_flat, dim=2, keepdim=True)[0]
        mad = torch.median(torch.abs(x_flat - median), dim=2, keepdim=True)[0]
        
        # Normalize
        normalized = (x_flat - median) / (mad + 1e-8)
        return normalized.view(b, c, h, w)
    
    def forward(self, x):
        # Block 1
        x1 = self.block1(x)
        if 1 in self.adain_blocks:
            x1 = self.adain_normalize(x1)
            
        # Block 2
        x2 = self.block2(x1)
        if 2 in self.adain_blocks:
            x2 = self.adain_normalize(x2)
            
        # Block 3
        x3 = self.block3(x2)
        
        return {'block1': x1, 'block2': x2, 'block3': x3}

class MemoryBank:
    """Stage-3: Memory bank with L2 distance (no faiss dependency)"""
    def __init__(self, distance_metric='l2', k=3):
        self.distance_metric = distance_metric
        self.k = k
        self.memory_features = None
        self.memory_size = 0
    
    def fit(self, features):
        """Fit memory bank with reference features"""
        if isinstance(features, torch.Tensor):
            # Cast to float32 first — BFloat16 is not numpy-compatible
            self.memory_features = features.detach().cpu().float().numpy()
        else:
            self.memory_features = np.array(features, dtype=np.float32)
        self.memory_size = len(self.memory_features)
        print(f"Memory bank fitted with {self.memory_size} features")

    def query(self, query_features, k=None):
        """Query k-nearest neighbors using L2 distance"""
        if k is None:
            k = self.k

        if isinstance(query_features, torch.Tensor):
            # Cast to float32 first — BFloat16 is not numpy-compatible
            query_features = query_features.detach().cpu().float().numpy()
        
        if self.memory_features is None:
            raise ValueError("Memory bank not fitted")
        
        # Compute L2 distances
        distances = cdist(query_features, self.memory_features, metric='euclidean')
        
        # Get k-nearest distances
        k_nearest_distances = np.sort(distances, axis=1)[:, :k]
        
        return k_nearest_distances

class DefectDetectionPipeline(nn.Module):
    """Complete 3-stage defect detection pipeline"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Stage 1: Coarse gate
        self.coarse_gate = MultiCueCoarseGate(
            acceptance_percentile=config.get('coarse_gate', {}).get('acceptance_percentile', 1.0)
        )

        # Stage 2: Frozen encoder
        encoder_config = config.get('encoder', {})
        self.encoder = FrozenEncoder(
            input_channels=encoder_config.get('input_channels', 3),
            adain_blocks=encoder_config.get('apply_adain', [1, 2])
        )

        # Stage 3: Memory bank
        memory_config = config.get('memory_bank', {})
        self.memory_bank = MemoryBank(
            distance_metric=memory_config.get('distance_metric', 'l2'),
            k=memory_config.get('search_k', 3)
        )

        # Learnable projection head: compresses block3 features (256*8*8=16384) → 128-dim
        # This is the only trainable component — used for feature normalization/compression
        feat_dim = 256 * 8 * 8  # block3 output: 256ch × 8×8 spatial
        proj_dim = config.get('projection_dim', 128)
        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )

        self.fitted = False
    
    def prepare_input(self, patches):
        """Prepare 3-channel input: grayscale + sobel-x + sobel-y"""
        if patches.shape[1] == 1:  # Already grayscale
            gray = patches
        else:
            gray = torch.mean(patches, dim=1, keepdim=True)
        
        # Sobel edges
        sobel_x = F.conv2d(gray, self.coarse_gate.sobel_x, padding=1)
        sobel_y = F.conv2d(gray, self.coarse_gate.sobel_y, padding=1)
        
        # Combine to 3-channel
        input_3ch = torch.cat([gray, sobel_x, sobel_y], dim=1)
        return input_3ch
    
    def fit(self, normal_patches):
        """Fit pipeline on normal samples"""
        self.eval()
        all_features = []
        
        with torch.no_grad():
            for batch in normal_patches:
                if isinstance(batch, (list, tuple)):
                    patches = batch[0]
                else:
                    patches = batch
                
                patches = patches.to(next(self.parameters()).device)
                
                # Prepare 3-channel input
                input_3ch = self.prepare_input(patches)
                
                # Extract features
                features = self.encoder(input_3ch)
                features_flat = features['block3'].view(features['block3'].shape[0], -1)
                all_features.append(features_flat.cpu())
        
        # Fit memory bank
        all_features = torch.cat(all_features, dim=0)
        self.memory_bank.fit(all_features)
        self.fitted = True
        
        print(f"Pipeline fitted on {len(all_features)} normal samples")
    
    def forward(self, patches):
        """Forward pass — returns projected features and anomaly scores.
        When memory bank is not yet fitted (smoke / train step), returns
        projected features only so that a reconstruction loss can be computed.
        """
        # Prepare 3-channel input
        input_3ch = self.prepare_input(patches)

        # Stage 1: Coarse gate (statistical cues, no grad)
        with torch.no_grad():
            cues = self.coarse_gate(patches)

        # Stage 2: Frozen feature extraction (no grad through encoder)
        with torch.no_grad():
            features = self.encoder(input_3ch)
            features_flat = features['block3'].view(features['block3'].shape[0], -1)

        # Learnable projection (gradients flow here)
        proj = self.projection_head(features_flat)

        if not self.fitted:
            # Pre-fitting mode: return projected features for loss computation
            return proj

        # Stage 3: Memory bank query (inference mode)
        distances = self.memory_bank.query(features_flat)
        anomaly_scores = np.mean(distances, axis=1)

        return {
            'cues': cues,
            'proj': proj,
            'features': features_flat,
            'anomaly_scores': torch.tensor(anomaly_scores, device=patches.device),
            'distances': distances,
        }

def build_model(config):
    """Build the defect detection model"""
    return DefectDetectionPipeline(config)