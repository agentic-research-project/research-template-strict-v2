import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
import numpy as np
from typing import Tuple, Dict, Any
from PIL import Image
import os

class SyntheticSemiconductorDataset(Dataset):
    """Synthetic semiconductor defect dataset for testing"""
    
    def __init__(self, num_samples: int = 1000, image_size: int = 6559, 
                 defect_ratio: float = 0.1, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.defect_ratio = defect_ratio
        self.transform = transform
        
        # Generate labels (0 = normal, 1 = defect)
        num_defects = int(num_samples * defect_ratio)
        self.labels = torch.cat([
            torch.zeros(num_samples - num_defects),
            torch.ones(num_defects)
        ]).long()
        
        # Shuffle labels
        indices = torch.randperm(num_samples)
        self.labels = self.labels[indices]
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = self.labels[idx].item()
        
        # Generate synthetic grayscale image
        # Base texture pattern
        np.random.seed(idx)  # Reproducible generation
        
        # Create base pattern with multiple frequency components on a reduced grid,
        # then resize to the target resolution to avoid extreme memory/time overhead.
        synth_size = min(self.image_size, 1024)
        x = np.linspace(0, 20*np.pi, synth_size)
        y = np.linspace(0, 20*np.pi, synth_size)
        X, Y = np.meshgrid(x, y)
        
        # Complex texture pattern
        base_pattern = (
            0.5 * np.sin(0.1*X) * np.cos(0.1*Y) +
            0.3 * np.sin(0.05*X + 0.3*Y) +
            0.2 * np.cos(0.15*X - 0.1*Y) +
            0.1 * np.random.normal(0, 0.1, (synth_size, synth_size))
        )
        if synth_size != self.image_size:
            base_pattern = np.array(
                Image.fromarray(base_pattern.astype(np.float32), mode='F').resize(
                    (self.image_size, self.image_size), Image.BILINEAR
                )
            )
        
        # Normalize to [0, 1]
        base_pattern = (base_pattern - base_pattern.min()) / (base_pattern.max() - base_pattern.min())
        
        if label == 1:  # Add defects
            # Add random defects
            num_defects = np.random.randint(1, 5)
            for _ in range(num_defects):
                # Random defect properties
                defect_y = np.random.randint(100, self.image_size - 100)
                defect_x = np.random.randint(100, self.image_size - 100)
                defect_size = np.random.randint(20, 100)
                defect_type = np.random.choice(['bright', 'dark', 'irregular'])
                
                y_start = max(0, defect_y - defect_size//2)
                y_end = min(self.image_size, defect_y + defect_size//2)
                x_start = max(0, defect_x - defect_size//2)
                x_end = min(self.image_size, defect_x + defect_size//2)
                
                if defect_type == 'bright':
                    base_pattern[y_start:y_end, x_start:x_end] = np.minimum(
                        base_pattern[y_start:y_end, x_start:x_end] + 0.3, 1.0
                    )
                elif defect_type == 'dark':
                    base_pattern[y_start:y_end, x_start:x_end] = np.maximum(
                        base_pattern[y_start:y_end, x_start:x_end] - 0.3, 0.0
                    )
                else:  # irregular
                    noise = np.random.normal(0, 0.2, (y_end-y_start, x_end-x_start))
                    base_pattern[y_start:y_end, x_start:x_end] += noise
                    base_pattern = np.clip(base_pattern, 0, 1)
        
        # Convert to tensor
        image = torch.from_numpy(base_pattern).float().unsqueeze(0)  # Add channel dim
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MVTecADSubset(Dataset):
    """Subset of MVTec AD dataset for method validation"""
    
    def __init__(self, root: str = './data/mvtec_ad', category: str = 'transistor', 
                 split: str = 'train', download: bool = True, transform=None):
        self.root = root
        self.category = category
        self.split = split
        self.transform = transform
        
        # Since MVTec AD might not be readily available, we'll use synthetic data
        # In a real implementation, this would load actual MVTec AD data
        print(f"Using synthetic MVTec-like data for category: {category}")
        
        if split == 'train':
            # Only normal samples in training
            self.dataset = SyntheticSemiconductorDataset(
                num_samples=200, image_size=512, defect_ratio=0.0, transform=transform
            )
        else:
            # Mixed normal and defect samples in test
            self.dataset = SyntheticSemiconductorDataset(
                num_samples=100, image_size=512, defect_ratio=0.3, transform=transform
            )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]

def create_transforms(image_size: int = 512, augment: bool = False) -> transforms.Compose:
    """Create image transforms"""
    transform_list = []
    
    # Resize to manageable size for testing (in practice, would use full 6559x6559)
    if image_size != 6559:
        transform_list.append(transforms.Resize((image_size, image_size)))
    
    if augment:
        # Style/noise robustness augmentations
        transform_list.extend([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.3)
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.3)
        ])
    
    # Normalize to [0, 1] range
    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    
    return transforms.Compose(transform_list)

def build_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders"""
    
    # Use smaller image size for testing (512 instead of 6559)
    # In production, would use full resolution
    test_image_size = config.get('image_size', 512)
    batch_size = config.get('batch_size', 16)
    num_workers = config.get('num_workers', 4)
    
    # Create transforms
    train_transform = create_transforms(image_size=test_image_size, augment=False)
    val_transform = create_transforms(image_size=test_image_size, augment=False)
    
    print(f"Creating synthetic semiconductor dataset (test size: {test_image_size}x{test_image_size})")
    print("In production, this would load actual 6559x6559 semiconductor images")
    
    # Create datasets
    # Training set: mostly normal samples for reference bank
    train_dataset = SyntheticSemiconductorDataset(
        num_samples=config.get('train_samples', 500),
        image_size=test_image_size,
        defect_ratio=0.05,  # Few defects in training
        transform=train_transform
    )
    
    # Validation set: mixed normal and defect samples
    val_dataset = SyntheticSemiconductorDataset(
        num_samples=config.get('val_samples', 200),
        image_size=test_image_size,
        defect_ratio=0.2,  # More defects for evaluation
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Process one image at a time for anomaly detection
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    return train_loader, val_loader