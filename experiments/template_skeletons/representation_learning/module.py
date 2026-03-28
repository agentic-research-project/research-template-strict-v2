"""Representation learning training module — linear probe eval."""
import torch, torch.nn as nn


class TrainingModule:
    def __init__(self, model, config):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        emb_dim = config.get("embedding_dim", 128)
        n_classes = config.get("num_classes", 10)
        self.linear_probe = nn.Linear(emb_dim, n_classes)

    def train_epoch(self, fabric, loader, optimizer):
        self.model.train()
        total = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            out = self.model(images)
            # supervised baseline: classify from embedding
            logits = self.linear_probe(out["embedding"])
            loss = self.criterion(logits, labels)
            fabric.backward(loss)
            optimizer.step()
            total += loss.item()
        return {"train_loss": total / len(loader)}

    def val_epoch(self, fabric, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                out = self.model(images)
                logits = self.linear_probe(out["embedding"])
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return {"linear_probe_accuracy": correct / max(total, 1)}
