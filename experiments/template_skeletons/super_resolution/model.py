"""Super-resolution model skeleton — sub-pixel upsampling."""
import torch.nn as nn


def build_model(config: dict) -> nn.Module:
    return SRNet(config)


class SRNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config.get("channels", 64)
        scale = config.get("upscale_factor", 4)
        in_ch = config.get("in_channels", 1)
        depth = config.get("depth", 4)

        body = [nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            body += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        body += [nn.Conv2d(ch, in_ch * scale * scale, 3, padding=1), nn.PixelShuffle(scale)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)
