"""Denoising model skeleton — residual noise prediction."""
import torch
import torch.nn as nn


def build_model(config: dict) -> nn.Module:
    return DenoisingNet(config)


class DenoisingNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config.get("channels", 64)
        depth = config.get("depth", 8)
        in_ch = config.get("in_channels", 1)

        layers = [nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(ch, in_ch, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.body(x)
        return x - noise  # residual learning: predict noise, subtract
