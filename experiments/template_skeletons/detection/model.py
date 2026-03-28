"""Detection model skeleton — anchor-free single-stage."""
import torch
import torch.nn as nn


def build_model(config: dict) -> nn.Module:
    return DetectionNet(config)


class DetectionNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config.get("channels", 64)
        n_classes = config.get("num_classes", 10)
        in_ch = config.get("in_channels", 3)

        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(ch * 4, n_classes, 1)
        self.reg_head = nn.Conv2d(ch * 4, 4, 1)  # xyxy offsets

    def forward(self, x):
        feat = self.backbone(x)
        cls_logits = self.cls_head(feat)
        bbox_pred = self.reg_head(feat)
        return {"cls_logits": cls_logits, "bbox_pred": bbox_pred}
