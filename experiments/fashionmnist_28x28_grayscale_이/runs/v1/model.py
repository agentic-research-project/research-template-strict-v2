import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.bn2(self.pointwise(x))
        return x


class ResidualDSBlock(nn.Module):
    """Residual block with depthwise-separable conv and SE attention."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, se_reduction: int):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_ch, out_ch, stride=stride, padding=1)
        self.se = SEBlock(out_ch, se_reduction)
        if in_ch == out_ch and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.se(self.conv(x)) + self.residual(x))


def build_model(config: dict) -> nn.Module:
    """Build lightweight CNN from config dict."""
    in_channels = config.get("in_channels", 1)
    channels = config.get("channels", [32, 64, 128])
    num_classes = config.get("num_classes", 10)
    se_reduction = config.get("se_reduction", 8)
    blocks_per_stage = config.get("blocks_per_stage", 2)

    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False),
        nn.BatchNorm2d(channels[0]),
        nn.ReLU(inplace=True),
    ]

    prev_ch = channels[0]
    for ch in channels:
        for blk_idx in range(blocks_per_stage):
            stride = 2 if (blk_idx == 0 and ch != prev_ch) else 1
            in_c = prev_ch if blk_idx == 0 else ch
            layers.append(ResidualDSBlock(in_c, ch, stride=stride, se_reduction=se_reduction))
        prev_ch = ch

    layers.extend([
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(channels[-1], num_classes),
    ])

    return nn.Sequential(*layers)
