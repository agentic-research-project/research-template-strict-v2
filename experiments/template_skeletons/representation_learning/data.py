"""Representation learning data skeleton."""
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_dataloaders(config: dict):
    n = config.get("n_samples", 500)
    bs = config.get("batch_size", 32)
    img_size = config.get("img_size", 32)
    in_ch = config.get("in_channels", 3)
    n_classes = config.get("num_classes", 10)

    images = torch.rand(n, in_ch, img_size, img_size)
    labels = torch.randint(0, n_classes, (n,))

    split = int(n * 0.8)
    train_ds = TensorDataset(images[:split], labels[:split])
    val_ds = TensorDataset(images[split:], labels[split:])
    return DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(val_ds, batch_size=bs)
