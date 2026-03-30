import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import Dict, List, Tuple
import cv2
from scipy import ndimage
from scipy.stats import genpareto
from sklearn.preprocessing import StandardScaler

class LoSAINModule(nn.Module):
    """Local Style-Adaptive Instance Normalization using retrieved reference statistics"""
    
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, query_features: torch.Tensor, reference_stats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply LoSAIN normalization
        Args:
            query_features: [B, C, H, W] query patch features
            reference_stats: dict with 'mean' [B, C, 1, 1] and 'std' [B, C, 1, 1] from retrieved references
        """
        # Compute query statistics
        query_mean = query_features.mean(dim=[2, 3], keepdim=True)
        query_std = query_features.std(dim=[2, 3], keepdim=True, unbiased=False) + self.eps
        
        # Normalize query features
        normalized = (query_features - query_mean) / query_std
        
        # Rescale to reference distribution
        ref_mean = reference_stats['mean']
        ref_std = reference_stats['std']
        
        return normalized * ref_std + ref_mean

class PatchExtractor:
    """Extract overlapping patches from high-resolution images"""
    
    def __init__(self, patch_size: int = 256, overlap: int = 32):
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
    
    def extract_patches(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Extract patches with coordinates
        Args:
            image: [1, 1, H, W] input image
        Returns:
            patches: [N, 1, patch_size, patch_size]
            coordinates: list of (y, x) top-left coordinates
        """
        _, _, H, W = image.shape
        patches = []
        coordinates = []
        
        y_positions = list(range(0, max(H - self.patch_size + 1, 1), self.stride))
        x_positions = list(range(0, max(W - self.patch_size + 1, 1), self.stride))
        if len(y_positions) == 0:
            y_positions = [0]
        if len(x_positions) == 0:
            x_positions = [0]
        
        # Ensure we cover the entire image
        if y_positions[-1] + self.patch_size < H:
            y_positions.append(H - self.patch_size)
        if x_positions[-1] + self.patch_size < W:
            x_positions.append(W - self.patch_size)
        
        for y in y_positions:
            for x in x_positions:
                patch = image[:, :, y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                coordinates.append((y, x))
        
        return torch.cat(patches, dim=0), coordinates
    
    def merge_patches(self, patch_scores: List[torch.Tensor], coordinates: List[Tuple[int, int]], 
                     image_shape: Tuple[int, int]) -> torch.Tensor:
        """Merge patch scores into full image anomaly map"""
        H, W = image_shape
        anomaly_map = torch.zeros((H, W))
        count_map = torch.zeros((H, W))
        
        for score, (y, x) in zip(patch_scores, coordinates):
            score_2d = score.view(self.patch_size, self.patch_size)
            anomaly_map[y:y+self.patch_size, x:x+self.patch_size] += score_2d
            count_map[y:y+self.patch_size, x:x+self.patch_size] += 1
        
        # Avoid division by zero
        count_map = torch.clamp(count_map, min=1)
        return anomaly_map / count_map

class FAISSIndex:
    """Simple FAISS-like functionality using PyTorch for k-NN search"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index_data = None
        self.is_trained = False
    
    def add(self, vectors: np.ndarray):
        """Add vectors to index"""
        if self.index_data is None:
            self.index_data = torch.from_numpy(vectors).float()
        else:
            new_vectors = torch.from_numpy(vectors).float()
            self.index_data = torch.cat([self.index_data, new_vectors], dim=0)
        self.is_trained = True
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors"""
        if not self.is_trained or self.index_data is None:
            raise RuntimeError("Index not trained or empty")
        
        query_tensor = torch.from_numpy(query).float()
        
        # Compute L2 distances
        distances = torch.cdist(query_tensor, self.index_data)
        
        # Get top-k
        top_distances, indices = torch.topk(distances, k, dim=1, largest=False)
        
        return top_distances.numpy(), indices.numpy()

def build_model(config: Dict) -> nn.Module:
    """Build anomaly detection model"""
    return DefectDetectionModel(config)

class DefectDetectionModel(nn.Module):
    """Training-free anomaly detection model using local reference retrieval"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Load pretrained backbone
        backbone_name = config.get('backbone', 'wide_resnet50_2')
        if backbone_name == 'wide_resnet50_2':
            self.backbone = models.wide_resnet50_2(pretrained=True)
        else:
            self.backbone = models.resnet50(pretrained=True)
        
        # Remove classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.eval()
        
        # Components
        self.patch_extractor = PatchExtractor(
            patch_size=config.get('patch_size', 256),
            overlap=config.get('patch_overlap', 32)
        )
        
        self.losain = LoSAINModule()
        
        # FAISS index for reference retrieval
        self.faiss_index = None
        self.reference_features = None
        self.reference_stats = None
        
        self.k = config.get('faiss_k', 8)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using pretrained backbone"""
        with torch.no_grad():
            features = self.backbone(x)
            return features
    
    def build_reference_bank(self, normal_images: List[torch.Tensor]):
        """Build reference bank from normal images"""
        all_features = []
        all_stats = []
        
        for image in normal_images:
            # Extract patches
            patches, _ = self.patch_extractor.extract_patches(image.unsqueeze(0))
            
            # Convert grayscale to RGB for backbone
            if patches.shape[1] == 1:
                patches = patches.repeat(1, 3, 1, 1)
            
            # Extract features
            features = self.extract_features(patches)
            
            # Flatten spatial dimensions
            B, C, H, W = features.shape
            features_flat = features.view(B, C * H * W)
            
            all_features.append(features_flat)
            
            # Compute and store statistics for LoSAIN
            feature_stats = {
                'mean': features.mean(dim=[2, 3], keepdim=True),
                'std': features.std(dim=[2, 3], keepdim=True, unbiased=False) + 1e-5
            }
            all_stats.append(feature_stats)
        
        # Concatenate all features
        self.reference_features = torch.cat(all_features, dim=0)
        self.reference_stats = {
            'mean': torch.cat([s['mean'] for s in all_stats], dim=0),
            'std': torch.cat([s['std'] for s in all_stats], dim=0)
        }
        
        # Build FAISS index
        feature_dim = self.reference_features.shape[1]
        self.faiss_index = FAISSIndex(feature_dim)
        self.faiss_index.add(self.reference_features.cpu().numpy())
        
        print(f"Built reference bank with {len(self.reference_features)} patches")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for anomaly detection"""
        if self.faiss_index is None:
            # Reference bank not yet built — return a zero anomaly map (used during
            # smoke-test train steps before build_reference_bank is called).
            H, W = x.shape[2], x.shape[3]
            return torch.zeros(x.shape[0], 1, H, W, device=x.device)
        
        # Extract patches
        patches, coordinates = self.patch_extractor.extract_patches(x)
        
        # Convert grayscale to RGB
        if patches.shape[1] == 1:
            patches = patches.repeat(1, 3, 1, 1)
        
        # Extract features
        query_features = self.extract_features(patches)
        
        # Flatten for FAISS search
        B, C, H, W = query_features.shape
        query_flat = query_features.view(B, C * H * W)
        
        # Retrieve nearest references
        distances, indices = self.faiss_index.search(query_flat.cpu().numpy(), self.k)
        
        # Compute anomaly scores (average distance to k-NN)
        anomaly_scores = torch.from_numpy(distances.mean(axis=1))
        
        # Merge patch scores into full image
        image_shape = x.shape[2:]
        anomaly_map = self.patch_extractor.merge_patches(
            [score.unsqueeze(0) for score in anomaly_scores], 
            coordinates, 
            image_shape
        )
        
        return anomaly_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]