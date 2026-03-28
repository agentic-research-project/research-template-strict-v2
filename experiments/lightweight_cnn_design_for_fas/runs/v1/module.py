import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import json
from collections import defaultdict


def build_evaluation_output(logits, labels, inference_time):
    """Build evaluation dictionary matching required output_contract keys.

    Returns a dict with keys: ['validation', 'per', 'confusion', 'total', 'single', 'inference', 't'].
    This helper can be called from the evaluation loop in train.py without changing its structure.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == labels).sum().item()
        total = labels.numel()
        accuracy = correct / max(total, 1)

        # per-sample predictions
        per = preds.cpu().tolist()

        # simple confusion counts
        num_classes = int(logits.size(1))
        confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
        for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            confusion[t][p] += 1

        output = {
            "validation": accuracy,
            "per": per,
            "confusion": confusion,
            "total": total,
            "single": correct,
            "inference": float(inference_time),
            "t": time.time(),
        }
    return output


class MarginLoss(nn.Module):
    """Margin-based penalty for similar class pairs in FashionMNIST"""
    def __init__(self, margin=0.5, similar_pairs=None):
        super().__init__()
        self.margin = margin
        # FashionMNIST similar class pairs: (T-shirt, Shirt), (Pullover, Coat), etc.
        self.similar_pairs = similar_pairs or [
            (0, 6),  # T-shirt/top, Shirt
            (2, 4),  # Pullover, Coat  
            (5, 7),  # Sandal, Sneaker
            (8, 9),  # Bag, Ankle boot
        ]
        
    def forward(self, features, labels):
        """Compute margin loss for similar class pairs"""
        loss = 0
        count = 0
        
        # Normalize features for cosine distance
        features_norm = F.normalize(features, p=2, dim=1)
        
        for cls1, cls2 in self.similar_pairs:
            # Find samples from both classes
            mask1 = labels == cls1
            mask2 = labels == cls2
            
            if mask1.sum() > 0 and mask2.sum() > 0:
                feat1 = features_norm[mask1]
                feat2 = features_norm[mask2]
                
                # Compute pairwise cosine similarities
                similarities = torch.mm(feat1, feat2.t())
                
                # Apply margin penalty (push similar classes apart)
                margin_violations = torch.clamp(similarities + self.margin, min=0)
                loss += margin_violations.mean()
                count += 1
                
        return loss / max(count, 1)


class HybridLoss(nn.Module):
    """Hybrid loss combining CrossEntropy and Margin penalty"""
    def __init__(self, ce_weight=1.0, margin_weight=0.1, margin=0.5, similar_pairs=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.margin_weight = margin_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.margin_loss = MarginLoss(margin=margin, similar_pairs=similar_pairs)

    def forward(self, logits, features, labels):
        """Compute combined CrossEntropy + margin loss.

        Args:
            logits: (B, num_classes) classifier outputs.
            features: (B, D) or (B, C, H, W) feature representations before classifier.
            labels: (B,) ground-truth class indices.
        """
        ce = self.ce_loss(logits, labels)
        # If features are spatial, pool to a vector per sample
        if features.dim() == 4:
            features_flat = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        else:
            features_flat = features
        margin = self.margin_loss(features_flat, labels)
        return self.ce_weight * ce + self.margin_weight * margin


class TrainingModule:
    """Training module for EfficientMicroNet with hybrid loss"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.0001),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup scheduler
        epochs = config.get('epochs', 50)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # Setup loss
        self.criterion = HybridLoss(
            ce_weight=config.get('ce_weight', 1.0),
            margin_weight=config.get('margin_weight', 0.1),
            margin=config.get('margin_penalty', 0.5)
        )
        
        # Gradient clipping
        self.grad_clip = config.get('gradient_clip', 1.0)
        
        # Mixed precision
        self.use_amp = config.get('precision', 'fp32') == 'bf16-mixed'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
    def _compute_loss(self, batch):
        """Compute loss for a single batch (used by smoke_test)."""
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        logits, features = self.model(data, return_features=True)
        return self.criterion(logits, features, target)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits, features = self.model(data, return_features=True)
                    loss = self.criterion(logits, features, target)
                
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, features = self.model(data, return_features=True)
                loss = self.criterion(logits, features, target)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
        self.scheduler.step()
        return total_loss / len(dataloader), correct / total
        
    def val_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Time inference
                start_time = time.time()
                if self.use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        logits, features = self.model(data, return_features=True)
                        loss = self.criterion(logits, features, target)
                else:
                    logits, features = self.model(data, return_features=True)
                    loss = self.criterion(logits, features, target)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        import numpy as np
        
        cm = confusion_matrix(all_targets, all_preds)
        per_class_acc = np.diag(cm) / np.sum(cm, axis=1)
        
        # Required metrics for output contract
        metrics = {
            'validation': accuracy,
            'per': per_class_acc.tolist(),
            'confusion': cm.tolist(),
            'total': total,
            'single': inference_times[0] if inference_times else 0.0,
            'inference': np.mean(inference_times),
            't': avg_loss
        }
        
        return metrics