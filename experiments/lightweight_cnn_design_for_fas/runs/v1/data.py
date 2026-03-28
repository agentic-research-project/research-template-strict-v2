import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os


def get_fashionmnist_transforms(is_train=True):
    """Get FashionMNIST data transforms with appropriate augmentation"""
    # FashionMNIST normalization values
    mean, std = 0.2860, 0.3530
    
    if is_train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])
    
    return transform


def create_dummy_data(config):
    """Create dummy data for testing when real data is not available"""
    print("Creating dummy FashionMNIST data for testing...")
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size, transform=None):
            self.size = size
            self.transform = transform
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Generate random 1x28x28 grayscale image to mimic FashionMNIST
            image = torch.randn(1, 28, 28)
            label = torch.randint(0, 10, (1,)).item()
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    batch_size = config.get('batch_size', 16)
    
    train_transform = get_fashionmnist_transforms(is_train=True)
    val_transform = get_fashionmnist_transforms(is_train=False)
    
    train_dataset = DummyDataset(1000, train_transform)
    val_dataset = DummyDataset(200, val_transform)
    test_dataset = DummyDataset(200, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


def build_dataloaders(config):
    """Build FashionMNIST dataloaders with train/val/test splits"""
    data_dir = config.get('data_dir', './data')
    batch_size = config.get('batch_size', 16)
    num_workers = config.get('num_workers', 2)
    val_split = config.get('val_split', 0.2)
    seed = config.get('seed', 42)
    
    # Set random seed for reproducible splits
    torch.manual_seed(seed)
    
    try:
        # Download FashionMNIST if not exists
        train_transform = get_fashionmnist_transforms(is_train=True)
        val_transform = get_fashionmnist_transforms(is_train=False)
        
        # Download training set
        full_train_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform
        )
        
        # Download test set
        test_dataset = datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=val_transform
        )
        
        # Split training set into train and validation
        total_size = len(full_train_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        # Update validation dataset transform
        val_dataset.dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=False,
            transform=val_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"Loaded FashionMNIST: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"Error loading FashionMNIST data: {e}")
        print("Falling back to dummy data...")
        return create_dummy_data(config)


def get_class_names():
    """Return FashionMNIST class names"""
    return [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]