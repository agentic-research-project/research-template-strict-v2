"""Super-resolution data skeleton — LR/HR pair handling."""
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


def build_dataloaders(config: dict):
    n = config.get("n_samples", 500)
    hr_size = config.get("hr_size", 64)
    scale = config.get("upscale_factor", 4)
    lr_size = hr_size // scale
    bs = config.get("batch_size", 16)
    in_ch = config.get("in_channels", 1)

    hr = torch.rand(n, in_ch, hr_size, hr_size)
    lr = F.interpolate(hr, size=(lr_size, lr_size), mode="bicubic", align_corners=False)

    split = int(n * 0.8)
    train_ds = TensorDataset(lr[:split], hr[:split])
    val_ds = TensorDataset(lr[split:], hr[split:])
    return DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(val_ds, batch_size=bs)
