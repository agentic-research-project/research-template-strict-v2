import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

def build_dataloaders(config):
    # Data path configuration
    data_dir = config.get('data_dir', '/data/0_Data/5_OpenSource/1_fashion_mnist')
    batch_size = config.get('batch_size', 16)
    num_workers = config.get('num_workers', 4)
    seed = config.get('seed', 42)
    
    # Set seed for reproducible splits
    torch.manual_seed(seed)
    
    # Data augmentation and normalization for FashionMNIST
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(28, padding=2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # FashionMNIST statistics
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    try:
        # Try to load from specified path first
        if os.path.exists(data_dir):
            train_dataset = datasets.FashionMNIST(
                root=data_dir,
                train=True,
                download=False,
                transform=train_transform
            )
            test_dataset = datasets.FashionMNIST(
                root=data_dir,
                train=False,
                download=False,
                transform=val_test_transform
            )
            print(f"Loaded FashionMNIST from {data_dir}")
        else:
            # Auto-download to default location if path doesn't exist
            print(f"Path {data_dir} not found, downloading FashionMNIST to ./data")
            train_dataset = datasets.FashionMNIST(
                root='./data',
                train=True,
                download=True,
                transform=train_transform
            )
            test_dataset = datasets.FashionMNIST(
                root='./data',
                train=False,
                download=True,
                transform=val_test_transform
            )
    except Exception as e:
        print(f"Warning: Could not load FashionMNIST dataset: {e}")
        print("Creating synthetic dummy data for testing...")
        
        # Create synthetic dummy data as fallback
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=1000, transform=None):
                self.size = size
                self.transform = transform
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                # Generate random 28x28 grayscale image
                image = torch.randn(1, 28, 28)
                label = torch.randint(0, 10, (1,)).item()
                
                if self.transform:
                    # Convert to PIL for transform compatibility
                    image = transforms.ToPILImage()(image.squeeze(0))
                    image = self.transform(image)
                
                return image, label
        
        train_dataset = DummyDataset(6000, train_transform)
        test_dataset = DummyDataset(1000, val_test_transform)
    
    # Create validation split from training data
    val_ratio = config.get('val_ratio', 0.1)
    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader