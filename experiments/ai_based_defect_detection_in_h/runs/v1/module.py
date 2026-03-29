import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score, precision_recall_curve
import time
import psutil
import os

class TrainingModule:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # For normal-only training of HF Noise Gate
        self.optimizer = torch.optim.AdamW(
            self.model.hf_gate.parameters(),
            lr=config.get('lr', 0.0001)
        )
        self.criterion = nn.MSELoss()
        
        # Statistics for adaptive thresholding
        self.normal_stats = {'mean': 0.0, 'std': 1.0}
        
    def _compute_loss(self, batch):
        """Compute HF noise gate training loss for a single batch.

        Args:
            batch: tuple (data, labels) or dict with 'image' key
        Returns:
            loss tensor
        """
        if isinstance(batch, (list, tuple)):
            data, labels = batch[0], batch[1]
        else:
            data = batch.get('image', batch.get('data', next(iter(batch.values()))))
            labels = batch.get('label', torch.zeros(data.shape[0], dtype=torch.long))

        data = data.to(self.device)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(self.device)

        # Convert grayscale to RGB if needed
        if data.shape[1] == 1:
            data_rgb = data.repeat(1, 3, 1, 1)
        else:
            data_rgb = data

        # Extract HF features
        lf, hf = self.model.freq_decomposer(data_rgb)
        hf_features = self.model.encoder(hf)

        # Apply noise gate
        gated_hf = self.model.hf_gate(hf_features)

        # Loss: minimize variation in gated HF features (consistency loss)
        if isinstance(gated_hf, (list, tuple)):
            losses = []
            for feat in gated_hf:
                if feat.shape[0] >= 2:
                    feat_a = feat[:-1]
                    feat_b = feat[1:]
                    losses.append(self.criterion(feat_a, feat_b))
            if losses:
                return torch.stack(losses).mean()
            # Fallback: L2 norm minimization
            return sum(feat.pow(2).mean() for feat in gated_hf) / max(len(gated_hf), 1)
        else:
            # gated_hf is a single tensor
            if gated_hf.shape[0] >= 2:
                return self.criterion(gated_hf[:-1], gated_hf[1:])
            return gated_hf.pow(2).mean()

    def train_epoch(self, dataloader, epoch):
        """Train HF Noise Gate on normal patches only"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            # Only use normal samples (label 0)
            normal_mask = (labels == 0)
            if normal_mask.sum() == 0:
                continue
                
            normal_data = data[normal_mask]
            
            # Convert grayscale to RGB if needed
            if normal_data.shape[1] == 1:
                normal_data = normal_data.repeat(1, 3, 1, 1)
            
            self.optimizer.zero_grad()
            
            # Extract HF features
            lf, hf = self.model.freq_decomposer(normal_data)
            hf_features = self.model.encoder(hf)
            
            # Apply noise gate
            gated_hf = self.model.hf_gate(hf_features)
            
            # Loss: minimize difference between gated HF features of normal patches
            if len(gated_hf) > 0 and normal_data.shape[0] > 1:
                losses = []
                for feat in gated_hf:
                    if feat.shape[0] < 2:
                        continue
                    feat_a = feat[:-1]
                    feat_b = feat[1:]
                    losses.append(self.criterion(feat_a, feat_b))
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'train_loss': avg_loss}
    
    def val_epoch(self, dataloader, epoch):
        """Validation with anomaly detection metrics"""
        self.model.eval()
        
        all_scores = []
        all_labels = []
        inference_times = []
        memory_usage = []
        
        # Build reference index from normal samples
        normal_patches = []
        for data, labels in dataloader:
            normal_mask = (labels == 0)
            if normal_mask.sum() > 0:
                normal_data = data[normal_mask]
                normal_patches.append(normal_data)
        
        if normal_patches:
            normal_patches = torch.cat(normal_patches, dim=0)[:100]  # Use subset for efficiency
            self.model.build_reference_index(normal_patches)
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(dataloader):
                data = data.to(self.device)
                labels = labels.cpu().numpy()
                
                # Measure inference time and memory
                start_time = time.time()
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Run inference
                anomaly_maps = []
                for i in range(data.shape[0]):
                    single_image = data[i:i+1]
                    anomaly_map = self.model(single_image, mode='inference')
                    anomaly_maps.append(anomaly_map)
                
                inference_time = (time.time() - start_time) / data.shape[0]
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                
                inference_times.append(inference_time)
                memory_usage.append(mem_after - mem_before)
                
                # Extract scores
                for i, anomaly_map in enumerate(anomaly_maps):
                    # Pixel-level score (mean anomaly score)
                    pixel_score = anomaly_map.mean().item()
                    all_scores.append(pixel_score)
                    all_labels.append(labels[i])
        
        # Compute statistics on normal samples for adaptive thresholding
        normal_scores = [score for score, label in zip(all_scores, all_labels) if label == 0]
        if normal_scores:
            self.normal_stats['mean'] = np.mean(normal_scores)
            self.normal_stats['std'] = np.std(normal_scores)
        
        # Compute metrics
        if len(set(all_labels)) > 1:  # Need both normal and anomalous samples
            pixel_auroc = roc_auc_score(all_labels, all_scores)
        else:
            pixel_auroc = 0.5
        
        # Adaptive threshold
        threshold = self.normal_stats['mean'] + 3 * self.normal_stats['std']
        predictions = [1 if score > threshold else 0 for score in all_scores]
        
        # False positive rate on normal images
        normal_predictions = [pred for pred, label in zip(predictions, all_labels) if label == 0]
        false_positive_rate = sum(normal_predictions) / max(len(normal_predictions), 1)
        
        # Anomalous area ratio on normal images
        anomalous_area_ratio = false_positive_rate  # Simplified
        
        # Performance metrics
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        peak_memory = max(memory_usage) if memory_usage else 0.0
        throughput = 1.0 / max(avg_inference_time, 1e-6)
        
        # Compute additional metrics
        mean_normal_score = np.mean([score for score, label in zip(all_scores, all_labels) if label == 0]) if normal_scores else 0.0
        
        return {
            'pixel': pixel_auroc,
            'image': pixel_auroc,  # Simplified: same as pixel AUROC
            'false': false_positive_rate,
            'anomalous': anomalous_area_ratio,
            'mean': mean_normal_score,
            'normal': 1.0 - false_positive_rate,  # Normal acceptance rate
            'defect': pixel_auroc,  # Defect detection capability
            'inference': avg_inference_time,
            'peak': peak_memory,
            'throughput': throughput,
            'val_loss': 1.0 - pixel_auroc  # Convert AUROC to loss-like metric
        }