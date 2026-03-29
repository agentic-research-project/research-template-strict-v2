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
        x = F.pad(x, (radius, radius, radius, radius), mode='reflect')
        return F.conv2d(x, kernel_2d, padding=0, groups=x.shape[1])

    def forward(self, x):
        # x: B, C, H, W
        lf = x
        for _ in range(max(1, self.levels)):
            lf = self._gaussian_blur(lf)
        hf = x - lf
        return lf, hf

class EfficientViTEncoder(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        # Use ResNet18 as lightweight alternative to EfficientViT-M4
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove avgpool and fc
        self.register_buffer('pixel_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.float()
        x = (x - self.pixel_mean) / self.pixel_std
        # Extract multi-scale features
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [5, 6, 7]:  # layer2, layer3, layer4 outputs
                features.append(F.normalize(x, p=2, dim=1))
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

    def forward(self, features):
        if isinstance(features, (list, tuple)):
            return [self._apply_gate(feat) for feat in features]
        else:
            return self._apply_gate(features)

    def _apply_gate(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        gate_key = str(C)
        if gate_key not in self._gates:
            # Fallback for unexpected channel sizes
            return x * 0.5  # Simple fixed gating
        
        # Spatial pooling to 1D
        x_pooled = F.adaptive_avg_pool2d(x, 1).view(B, C, 1)  # [B, C, 1]
        gate_weights = self._gates[gate_key](x_pooled)  # [B, C, 1]
        gate_weights = gate_weights.view(B, C, 1, 1)  # [B, C, 1, 1]
        
        return x * gate_weights

class FreqConsensusAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.freq_decomposer = GaussianPyramid(
            sigma=config.get('freq_sigma', 2.0),
            levels=config.get('freq_levels', 2)
        )
        self.encoder = EfficientViTEncoder(frozen=True)
        self.hf_gate = HFNoiseGate(channels=512)
        
    def forward(self, x):
        # Frequency decomposition
        lf, hf = self.freq_decomposer(x)
        
        # Extract features
        hf_features = self.encoder(hf)
        
        # Apply noise gate
        gated_hf = self.hf_gate(hf_features)
        
        return gated_hf

def build_model(config):
    """Build and return the FreqConsensusAD model"""
    return FreqConsensusAD(config)