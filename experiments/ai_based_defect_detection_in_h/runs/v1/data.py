import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
from PIL import Image
import os

class SyntheticSemiconductorDataset(Dataset):
    """Synthetic dataset mimicking semiconductor defect detection"""
    
    def __init__(self, num_samples=1000, image_size=512, split='train', transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.split = split
        self.transform = transform
        
        # 80% normal, 20% defective for train; 70% normal, 30% defective for val
        if split == 'train':
            self.defect_ratio = 0.2
        else:
            self.defect_ratio = 0.3
            
    def __len__(self):
        return self.num_samples
    
    def generate_normal_pattern(self):
        """Generate normal semiconductor-like pattern"""
        # Create regular grid pattern
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Add periodic structure (simulating circuit patterns)
        period = 32
        for i in range(0, self.image_size, period):
            for j in range(0, self.image_size, period):
                # Add rectangular pattern
                img[i:i+period//2, j:j+period//2] = 0.3
                img[i+period//4:i+3*period//4, j+period//4:j+3*period//4] = 0.7
        
        # Add some texture
        noise = np.random.normal(0, 0.05, (self.image_size, self.image_size))
        img = np.clip(img + noise, 0, 1)
        
        return img
    
    def add_defects(self, img):
        """Add various types of defects to the image"""
        img_defective = img.copy()
        
        # Random number of defects (1-5)
        num_defects = np.random.randint(1, 6)
        
        for _ in range(num_defects):
            defect_type = np.random.choice(['scratch', 'particle', 'stain', 'bridge'])
            x = np.random.randint(0, self.image_size)
            y = np.random.randint(0, self.image_size)
            r = np.random.randint(3, 12)
            if defect_type == 'scratch':
                img_defective[max(0, y - 1):min(self.image_size, y + 2), max(0, x - 4 * r):min(self.image_size, x + 4 * r)] = 1.0
            elif defect_type == 'particle':
                yy, xx = np.ogrid[:self.image_size, :self.image_size]
                mask = (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2
                img_defective[mask] = 1.0
            elif defect_type == 'stain':
                img_defective[max(0, y - r):min(self.image_size, y + r), max(0, x - r):min(self.image_size, x + r)] *= 0.3
            else:  # bridge
                img_defective[max(0, y - r):min(self.image_size, y + r), max(0, x - 2 * r):min(self.image_size, x + 2 * r)] = 0.8
        return np.clip(img_defective, 0, 1)
    
    def __getitem__(self, idx):
        # Generate base pattern
        img = self.generate_normal_pattern()
        
        # Determine if this sample should have defects
        is_defective = np.random.random() < self.defect_ratio
        
        if is_defective:
            img = self.add_defects(img)
            label = 1  # Anomalous
        else:
            label = 0  # Normal
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(img).unsqueeze(0)  # Add channel dimension
        
        # Apply transforms if provided
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, label

def build_dataloaders(config):
    """Build data loaders for semiconductor defect detection"""
    
    # Extract configuration
    batch_size = config.get('batch_size', 16)
    num_workers = config.get('num_workers', 4)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize for efficiency
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    # Try to load real data first, fallback to synthetic
    data_path = config.get('data_path', '')
    
    if data_path and os.path.exists(data_path):
        # Real data loading (placeholder - would need actual implementation)
        print(f"Loading data from {data_path}")
        # This would be implemented based on actual data structure
        train_dataset = SyntheticSemiconductorDataset(
            num_samples=800, split='train', transform=transform
        )
        val_dataset = SyntheticSemiconductorDataset(
            num_samples=200, split='val', transform=transform
        )
    else:
        # Fallback to synthetic data
        print("Using synthetic semiconductor data")
        train_dataset = SyntheticSemiconductorDataset(
            num_samples=800, split='train', transform=transform
        )
        val_dataset = SyntheticSemiconductorDataset(
            num_samples=200, split='val', transform=transform
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
    
    return train_loader, val_loader