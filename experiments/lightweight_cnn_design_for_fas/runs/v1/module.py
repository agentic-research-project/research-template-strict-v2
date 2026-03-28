"""
module.py — TrainingModule for PyTorch Fabric (non-Lightning interface)

Provides train_epoch / val_epoch methods compatible with the train.py template.
Outputs METRICS:{json} to stdout as required by the output contract.
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainingModule:
    """Wraps a model and provides train_epoch / val_epoch for Fabric."""

    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

    def _compute_loss(self, batch):
        """Compute loss for a single batch. Used by smoke_test."""
        x, y = batch
        logits = self.model(x)
        return self.criterion(logits, y)

    def train_epoch(self, fabric, train_loader, optimizer):
        """One training epoch. Returns dict of metrics."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        gradient_clip = self.config.get("gradient_clip", None)

        for batch in train_loader:
            x, y = batch
            optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            fabric.backward(loss)
            if gradient_clip:
                fabric.clip_gradients(self.model, optimizer, max_norm=gradient_clip)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }

    def val_epoch(self, fabric, val_loader):
        """One validation epoch. Returns dict of metrics."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            x, y = batch
            logits = self.model(x)
            loss = self.criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
        }
