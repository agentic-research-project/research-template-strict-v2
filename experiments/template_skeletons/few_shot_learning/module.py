"""Few-shot learning training module — episodic training."""
import torch, torch.nn as nn


class TrainingModule:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, fabric, loader, optimizer):
        self.model.train()
        total_loss, n = 0.0, 0
        for batch in loader:
            support_x, support_y, query_x, query_y = [b.squeeze(0) for b in batch]
            optimizer.zero_grad()
            prototypes = self.model.compute_prototypes(support_x, support_y, self.config.get("n_way", 5))
            logits = self.model.classify(query_x, prototypes)
            loss = self.criterion(logits, query_y)
            fabric.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        return {"train_loss": total_loss / max(n, 1)}

    def val_epoch(self, fabric, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                support_x, support_y, query_x, query_y = [b.squeeze(0) for b in batch]
                prototypes = self.model.compute_prototypes(support_x, support_y, self.config.get("n_way", 5))
                logits = self.model.classify(query_x, prototypes)
                correct += (logits.argmax(1) == query_y).sum().item()
                total += query_y.size(0)
        return {"few_shot_accuracy": correct / max(total, 1)}
