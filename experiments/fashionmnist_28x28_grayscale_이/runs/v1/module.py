"""
module.py — TrainingModule: train/val epoch 로직
train.py의 인터페이스에 맞춤: TrainingModule(model, config)
"""
import json
import torch
import torch.nn as nn


class TrainingModule:
    """Training and validation logic for FashionMNIST CNN."""

    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, fabric, train_loader, optimizer) -> dict:
        """Run one training epoch. Returns dict with 'loss' and 'accuracy'."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            fabric.backward(loss)

            # Gradient clipping
            clip_val = self.config.get("gradient_clip", 1.0)
            if clip_val:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)

            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        return {"loss": avg_loss, "accuracy": accuracy}

    @torch.no_grad()
    def val_epoch(self, fabric, val_loader) -> dict:
        """Run one validation epoch. Returns dict with 'accuracy' and 'loss'.
        Prints METRICS:{json} for automated parsing."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in val_loader:
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        metrics = {"accuracy": accuracy, "loss": avg_loss}
        print(f"METRICS:{json.dumps(metrics)}")

        return metrics
