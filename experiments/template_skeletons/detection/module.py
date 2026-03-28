"""Detection training module — mAP placeholder."""
import torch, torch.nn as nn


class TrainingModule:
    def __init__(self, model, config):
        self.model = model
        self.cls_loss = nn.CrossEntropyLoss()

    def train_epoch(self, fabric, loader, optimizer):
        self.model.train()
        total = 0.0
        for images, bboxes, labels in loader:
            optimizer.zero_grad()
            out = self.model(images)
            # 간소화: cls loss만 (bbox loss는 구현 필요)
            B = out["cls_logits"].shape[0]
            cls = out["cls_logits"].mean(dim=[2, 3])  # spatial avg
            loss = self.cls_loss(cls, labels.squeeze(1))
            fabric.backward(loss)
            optimizer.step()
            total += loss.item()
        return {"train_loss": total / len(loader)}

    def val_epoch(self, fabric, loader):
        self.model.eval()
        # mAP placeholder — 실제 구현 시 COCO eval 연동 필요
        return {"mAP": 0.0, "AP50": 0.0}
