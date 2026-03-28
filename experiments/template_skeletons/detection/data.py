"""Detection data skeleton — bbox/class pair handling."""
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_dataloaders(config: dict):
    n = config.get("n_samples", 200)
    bs = config.get("batch_size", 8)
    img_size = config.get("img_size", 128)
    in_ch = config.get("in_channels", 3)
    n_classes = config.get("num_classes", 10)

    images = torch.rand(n, in_ch, img_size, img_size)
    # 더미 bbox targets (배치별 고정 1개 bbox)
    bboxes = torch.rand(n, 1, 4) * img_size  # xyxy
    labels = torch.randint(0, n_classes, (n, 1))

    split = int(n * 0.8)
    train_ds = TensorDataset(images[:split], bboxes[:split], labels[:split])
    val_ds = TensorDataset(images[split:], bboxes[split:], labels[split:])
    return DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(val_ds, batch_size=bs)
