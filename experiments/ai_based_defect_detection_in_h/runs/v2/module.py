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

        # Keep decomposition in native grayscale space; encoder handles RGB expansion if needed
        lf, hf = self.model.freq_decomposer(data)
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
                    loss = self.criterion(feat_a, feat_b)
                    losses.append(loss)
                else:
                    losses.append(torch.tensor(0.0, device=self.device))
            total_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0, device=self.device)
        else:
            if gated_hf.shape[0] >= 2:
                feat_a = gated_hf[:-1]
                feat_b = gated_hf[1:]
                total_loss = self.criterion(feat_a, feat_b)
            else:
                total_loss = torch.tensor(0.0, device=self.device)

        return total_loss

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / max(num_batches, 1)

    def evaluate(self, dataloader):
        """Evaluate model and compute pixel-level AUROC"""
        self.model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data, labels = batch[0], batch[1]
                else:
                    data = batch.get('image', batch.get('data', next(iter(batch.values()))))
                    labels = batch.get('label', torch.zeros(data.shape[0], dtype=torch.long))
                
                data = data.to(self.device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)
                
                # Get anomaly scores (simple implementation for now)
                gated_hf = self.model(data)
                
                # Compute pixel-level anomaly scores
                if isinstance(gated_hf, (list, tuple)):
                    # Use the last scale features
                    feat = gated_hf[-1]
                else:
                    feat = gated_hf
                
                # Upsample to original size and compute scores
                B, C, H, W = feat.shape
                target_size = data.shape[-2:]
                
                # Simple anomaly scoring: use feature magnitude
                scores = torch.norm(feat, dim=1, keepdim=True)  # [B, 1, H, W]
                scores = F.interpolate(scores, size=target_size, mode='bilinear', align_corners=False)
                
                # Create pixel-level labels (0=normal, 1=anomaly)
                if labels.dim() == 1:  # image-level labels
                    pixel_labels = labels.view(-1, 1, 1, 1).expand(-1, 1, *target_size)
                else:
                    pixel_labels = labels
                
                all_scores.append(scores.cpu().numpy())
                all_labels.append(pixel_labels.cpu().numpy())
        
        # Concatenate all results
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Flatten for pixel-level evaluation
        scores_flat = all_scores.flatten()
        labels_flat = all_labels.flatten()
        
        # Compute pixel-level AUROC
        try:
            pixel_auroc = roc_auc_score(labels_flat, scores_flat)
        except ValueError:
            # Handle case where all labels are the same
            pixel_auroc = 0.5
        
        return {
            'pixel_auroc': pixel_auroc,
            'loss': 0.0
        }