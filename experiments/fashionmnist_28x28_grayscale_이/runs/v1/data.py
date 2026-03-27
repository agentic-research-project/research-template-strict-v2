"""
data.py — FashionMNIST 데이터 로더
torchvision으로 자동 다운로드, train/val split, augmentation 적용
"""
import os

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) from config."""
    data_dir = config.get("data_dir", "/data/0_Data/5_OpenSource/1_fashion_mnist")
    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 4)
    val_ratio = config.get("val_ratio", 0.1)
    seed = config.get("seed", 42)

    os.makedirs(data_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    # Download FashionMNIST
    full_train_aug = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=train_transform,
    )
    full_train_val = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=val_transform,
    )

    # Split indices
    n_total = len(full_train_aug)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(full_train_aug, train_indices)
    val_dataset = Subset(full_train_val, val_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
