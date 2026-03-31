import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional
import cv2
from scipy import stats
from scipy.stats import genpareto
from sklearn.preprocessing import StandardScaler

class FaissReferenceBank:
    """FAISS-backed normal reference bank for top-k retrieval."""
    def __init__(self, dim: int, nlist: int = 100, m: int = 8, nbits: int = 8, use_gpu: bool = False):
        self.dim = dim
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.index = None

    def build(self, features: np.ndarray):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.dim:
            raise ValueError(f"Expected features of shape (N, {self.dim}), got {features.shape}")
        if features.shape[0] >= max(self.nlist * 4, 256):
            quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFPQ(quantizer, self.dim, self.nlist, self.m, self.nbits)
            index.train(features)
            index.add(features)
            index.nprobe = min(16, self.nlist)
        else:
            index = faiss.IndexFlatL2(self.dim)
            index.add(features)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        self.index = index

    def search(self, queries: np.ndarray, k: int = 8):
        if self.index is None:
            raise RuntimeError("Reference bank has not been built")
        queries = np.asarray(queries, dtype=np.float32)
        return self.index.search(queries, k)

class WideResNetFeatureExtractor(nn.Module):
    """Frozen WideResNet-50 feature extractor adapted for grayscale input"""
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pretrained WideResNet-50
        try:
            weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.wide_resnet50_2(weights=weights)
        except AttributeError:
            self.backbone = models.wide_resnet50_2(pretrained=pretrained)
        
        # Adapt first conv for grayscale by averaging RGB weights
        if pretrained:
            conv1_weight = self.backbone.conv1.weight.data
            # Average across RGB channels for grayscale adaptation
            gray_weight = conv1_weight.mean(dim=1, keepdim=True)
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone.conv1.weight.data = gray_weight
        else:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove final layers (we only need intermediate features)
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, x):
        """Extract multi-scale features from layers 2 and 3"""
        features = {}
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        
        x = self.backbone.layer2(x)
        features['layer2'] = x  # 32x32 for 256x256 input
        
        x = self.backbone.layer3(x)
        features['layer3'] = x  # 16x16 for 256x256 input
        
        return features

class LoSAINModule(nn.Module):
    """Local Style-Adaptive Instance Normalization using retrieved reference statistics"""
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, query_features, reference_features_list):
        """
        Apply LoSAIN normalization
        Args:
            query_features: (B, C, H, W) query patch features
            reference_features_list: List of (C, H, W) reference features
        Returns:
            normalized_features: (B, C, H, W) LoSAIN normalized features
        """
        B, C, H, W = query_features.shape
        
        # Compute query statistics
        query_mean = query_features.view(B, C, -1).mean(dim=2, keepdim=True)
        query_var = query_features.view(B, C, -1).var(dim=2, keepdim=True, unbiased=False)
        query_std = torch.sqrt(query_var + self.eps)
        
        # Compute reference statistics from retrieved top-k references
        if isinstance(reference_features_list, torch.Tensor):
            # Expected shape: (B, K, C, H, W)
            ref = reference_features_list
            ref_mean = ref.view(ref.shape[0], ref.shape[1], C, -1).mean(dim=3)
            ref_std = torch.sqrt(ref.view(ref.shape[0], ref.shape[1], C, -1).var(dim=3, unbiased=False) + self.eps)
            mu_ref = ref_mean.mean(dim=1).view(B, C, 1, 1)
            sigma_ref = ref_std.mean(dim=1).view(B, C, 1, 1)
        else:
            # Fallback: list of reference tensors, each shaped (C, H, W) or (1, C, H, W)
            ref_means = []
            ref_stds = []
            for ref in reference_features_list:
                if ref.dim() == 4:
                    ref = ref.squeeze(0)
                ref_flat = ref.view(C, -1)
                ref_means.append(ref_flat.mean(dim=1))
                ref_stds.append(torch.sqrt(ref_flat.var(dim=1, unbiased=False) + self.eps))
            mu_ref = torch.stack(ref_means, dim=0).mean(dim=0).view(1, C, 1, 1).expand(B, -1, -1, -1)
            sigma_ref = torch.stack(ref_stds, dim=0).mean(dim=0).view(1, C, 1, 1).expand(B, -1, -1, -1)

        query_mean = query_mean.view(B, C, 1, 1)
        query_std = query_std.view(B, C, 1, 1)
        normalized_features = sigma_ref * (query_features - query_mean) / query_std + mu_ref
        return normalized_features

class DefectDetectionModel(nn.Module):
    """Main defect detection model implementing the full pipeline"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config.get('patch_size', 256)
        self.overlap = config.get('patch_overlap', 32)
        self.k_references = config.get('k_references', 8)
        self.target_fpr = config.get('target_fpr', 0.01)
        
        # Feature extractor
        self.feature_extractor = WideResNetFeatureExtractor(pretrained=True)

        # LoSAIN module
        self.losain = LoSAINModule()

        # Reference bank (will be populated during setup)
        self.reference_bank = {}
        self.faiss_indices = {}
        self.evt_thresholds = {}

        # Dummy learnable parameter (for optimizer compatibility — model is otherwise frozen)
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)
        
    def extract_patches(self, image):
        """Extract overlapping patches from high-resolution image"""
        H, W = image.shape[-2:]
        patches = []
        positions = []
        
        step = self.patch_size - self.overlap
        
        for y in range(0, H - self.patch_size + 1, step):
            for x in range(0, W - self.patch_size + 1, step):
                patch = image[..., y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                positions.append((y, x))
        
        return torch.stack(patches), positions
    
    def build_reference_bank(self, normal_patches, condition_id='default'):
        """Build FAISS index for normal reference patches"""
        reference_features = {'layer2': [], 'layer3': []}
        
        with torch.no_grad():
            for patch in normal_patches:
                if patch.dim() == 3:
                    patch = patch.unsqueeze(0)
                features = self.feature_extractor(patch)
                
                for layer_name in ['layer2', 'layer3']:
                    feat = features[layer_name].squeeze(0).cpu().numpy()
                    reference_features[layer_name].append(feat.flatten())
        
        # Build FAISS indices for each layer
        self.faiss_indices[condition_id] = {}
        self.reference_bank[condition_id] = {}
        
        for layer_name in ['layer2', 'layer3']:
            ref_array = np.array(reference_features[layer_name]).astype(np.float32)
            self.reference_bank[condition_id][layer_name] = ref_array
            
            # Build FAISS IVF-PQ index
            d = ref_array.shape[1]
            nlist = min(100, len(ref_array) // 10)  # Number of clusters
            m = min(64, d // 8)  # Number of subquantizers
            
            if len(ref_array) > 1000:
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                index.train(ref_array)
                index.add(ref_array)
            else:
                # Use simpler index for small datasets
                index = faiss.IndexFlatL2(d)
                index.add(ref_array)
            
            self.faiss_indices[condition_id][layer_name] = index
    
    def retrieve_references(self, query_features, condition_id='default', k=None):
        """Retrieve top-k nearest references using FAISS"""
        if k is None:
            k = self.k_references
        
        retrieved_refs = {}
        
        for layer_name in ['layer2', 'layer3']:
            query_feat = query_features[layer_name].cpu().numpy().flatten().astype(np.float32)
            query_feat = query_feat.reshape(1, -1)
            
            # Search in FAISS index
            distances, indices = self.faiss_indices[condition_id][layer_name].search(query_feat, k)
            
            # Get reference features
            ref_feats = []
            for idx in indices[0]:
                ref_feat = self.reference_bank[condition_id][layer_name][idx]
                original_shape = query_features[layer_name].shape
                ref_feat = torch.from_numpy(ref_feat.reshape(original_shape))
                ref_feats.append(ref_feat)
            
            retrieved_refs[layer_name] = ref_feats
        
        return retrieved_refs
    
    def compute_anomaly_score(self, query_features, retrieved_refs):
        """Compute multi-scale anomaly score after LoSAIN normalization"""
        scores = []
        
        for layer_name in ['layer2', 'layer3']:
            query_feat = query_features[layer_name]
            ref_feats = retrieved_refs[layer_name]
            
            # Apply LoSAIN normalization
            # query_feat is [1, C, H_feat, W_feat] (batch of 1); pass directly (no unsqueeze)
            normalized_query = self.losain(query_feat, ref_feats)
            normalized_query = normalized_query.squeeze(0)  # [C, H_feat, W_feat]

            # Compute minimum L2 distance to references
            min_dist = float('inf')
            for ref_feat in ref_feats:
                # ref_feat from retrieve_references is reshaped to match query_features shape
                dist = torch.norm(normalized_query - ref_feat.squeeze(0), p=2).item()
                min_dist = min(min_dist, dist)
            
            scores.append(min_dist)
        
        # Equal weighting across layers
        return np.mean(scores)
    
    def merge_patch_scores(self, patch_scores, positions, image_shape):
        """Merge patch-level scores into full-resolution anomaly map"""
        H, W = image_shape
        anomaly_map = np.zeros((H, W))
        weight_map = np.zeros((H, W))
        
        # Create Gaussian weight kernel
        kernel_size = self.patch_size
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                kernel[i, j] = np.exp(-(dist**2) / (2 * (center/3)**2))
        
        # Blend patches with Gaussian weighting
        for score, (y, x) in zip(patch_scores, positions):
            end_y, end_x = y + self.patch_size, x + self.patch_size
            anomaly_map[y:end_y, x:end_x] += score * kernel
            weight_map[y:end_y, x:end_x] += kernel
        
        # Normalize by weights
        anomaly_map = np.divide(anomaly_map, weight_map, 
                               out=np.zeros_like(anomaly_map), 
                               where=weight_map!=0)
        
        return anomaly_map
    
    def fit_evt_threshold(self, normal_scores, condition_id='default', percentile=95):
        """Fit EVT (Generalized Pareto Distribution) for threshold calibration"""
        # Use top percentile for EVT fitting
        threshold_init = np.percentile(normal_scores, percentile)
        excesses = normal_scores[normal_scores > threshold_init] - threshold_init
        
        if len(excesses) < 10:
            # Fallback to percentile-based threshold
            self.evt_thresholds[condition_id] = np.percentile(normal_scores, 
                                                             (1-self.target_fpr)*100)
            return
        
        try:
            # Fit GPD to excesses
            params = genpareto.fit(excesses)
            shape, loc, scale = params
            
            # Compute threshold for target FPR
            if shape != 0:
                threshold = threshold_init + (scale/shape) * ((self.target_fpr * len(normal_scores) / len(excesses))**(-shape) - 1)
            else:
                threshold = threshold_init - scale * np.log(self.target_fpr * len(normal_scores) / len(excesses))
            
            self.evt_thresholds[condition_id] = max(threshold, threshold_init)
            
        except:
            # Fallback to percentile-based threshold
            self.evt_thresholds[condition_id] = np.percentile(normal_scores, 
                                                             (1-self.target_fpr)*100)
    
    def postprocess_anomaly_map(self, anomaly_map, condition_id='default'):
        """Apply morphological operations and connected component analysis"""
        threshold = self.evt_thresholds.get(condition_id, np.percentile(anomaly_map, 99))
        
        # Threshold anomaly map
        binary_map = (anomaly_map > threshold).astype(np.uint8)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_OPENING, kernel)
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSING, kernel)
        
        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map)
        
        # Filter by minimum area
        min_area = self.config.get('min_defect_area', 100)
        filtered_labels = np.zeros_like(labels)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_labels[labels == i] = i
        
        return anomaly_map, filtered_labels, threshold
    
    def forward(self, x, condition_id='default', return_details=False):
        """Forward pass for anomaly detection."""
        H = x.shape[-2] if x.dim() >= 3 else self.patch_size
        W = x.shape[-1] if x.dim() >= 3 else self.patch_size

        # If reference bank not yet built, return zero anomaly map (smoke-test / cold start)
        if condition_id not in self.faiss_indices:
            zero_map = torch.zeros(H, W, dtype=torch.float32)
            if return_details:
                return zero_map, torch.zeros_like(zero_map), torch.zeros_like(zero_map), 0.0
            return zero_map

        # Extract patches
        patches, positions = self.extract_patches(x)

        patch_scores = []

        # Process each patch
        for patch in patches:
            if patch.dim() == 3:
                patch = patch.unsqueeze(0)

            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(patch)

            # Retrieve references
            retrieved_refs = self.retrieve_references(features, condition_id)

            # Compute anomaly score
            score = self.compute_anomaly_score(features, retrieved_refs)
            patch_scores.append(score)

        # Merge patch scores into full anomaly map
        image_shape = x.shape[-2:]
        anomaly_map_np = self.merge_patch_scores(patch_scores, positions, image_shape)
        anomaly_map = torch.from_numpy(anomaly_map_np.astype(np.float32))

        if return_details:
            processed_map_np, labels_np, threshold = self.postprocess_anomaly_map(anomaly_map_np, condition_id)
            return (anomaly_map,
                    torch.from_numpy(processed_map_np.astype(np.float32)),
                    torch.from_numpy(labels_np.astype(np.int32)),
                    threshold)

        return anomaly_map

def build_model(config):
    """Build the defect detection model"""
    return DefectDetectionModel(config)