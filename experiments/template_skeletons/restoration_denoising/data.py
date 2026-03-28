"""Denoising data skeleton — noisy/clean pair handling."""
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_dataloaders(config: dict):
    """noisy/clean pair DataLoader. data_dir에 데이터가 있으면 로드, 없으면 합성."""
    from pathlib import Path
    data_dir = config.get("data_dir", "")
    batch_size = config.get("batch_size", 16)
    n_samples = config.get("n_samples", 1000)
    img_size = config.get("img_size", 64)
    in_ch = config.get("in_channels", 1)
    noise_std = config.get("noise_std", 0.1)

    if data_dir and Path(data_dir).exists() and any(Path(data_dir).iterdir()):
        # 실제 데이터 로드 (구현 필요)
        raise NotImplementedError(f"Load real data from {data_dir}")

    # 합성 데이터 fallback
    clean = torch.rand(n_samples, in_ch, img_size, img_size)
    noisy = clean + torch.randn_like(clean) * noise_std
    noisy = noisy.clamp(0, 1)

    split = int(n_samples * 0.8)
    train_ds = TensorDataset(noisy[:split], clean[:split])
    val_ds = TensorDataset(noisy[split:], clean[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
