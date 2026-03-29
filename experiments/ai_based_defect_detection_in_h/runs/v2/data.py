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
            x = np.random.randint(50, self.image_size - 50)
            y = np.random.randint(50, self.image_size - 50)
            
            if defect_type == 'scratch':
                # Linear defect
                length = np.random.randint(20, 100)
                angle = np.random.uniform(0, 2 * np.pi)
                dx = int(length * np.cos(angle))
                dy = int(length * np.sin(angle))
                
                # Draw line
                for i in range(length):
                    px = x + int(i * dx / length)
                    py = y + int(i * dy / length)
                    if 0 <= px < self.image_size and 0 <= py < self.image_size:
                        img_defective[py, px] = np.random.uniform(0.1, 0.9)
                        
            elif defect_type == 'particle':
                # Circular defect
                radius = np.random.randint(3, 15)
                intensity = np.random.uniform(0.2, 0.8)
                
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx*dx + dy*dy <= radius*radius:
                            px, py = x + dx, y + dy
                            if 0 <= px < self.image_size and 0 <= py < self.image_size:
                                img_defective[py, px] = intensity
                                
            elif defect_type == 'stain':
                # Irregular stain
                size = np.random.randint(10, 30)
                intensity = np.random.uniform(0.1, 0.9)
                
                for _ in range(size * 2):
                    dx = np.random.randint(-size, size)
                    dy = np.random.randint(-size, size)
                    px, py = x + dx, y + dy
                    if 0 <= px < self.image_size and 0 <= py < self.image_size:
                        img_defective[py, px] = intensity
                        
            elif defect_type == 'bridge':
                # Bridge connecting two points
                width = np.random.randint(2, 8)
                length = np.random.randint(15, 40)
                
                for i in range(length):
                    for j in range(width):
                        px = x + i
                        py = y + j
                        if 0 <= px < self.image_size and 0 <= py < self.image_size:
                            img_defective[py, px] = np.random.uniform(0.3, 0.7)
        
        return img_defective
    
    def __getitem__(self, idx):
        # Generate base pattern
        img = self.generate_normal_pattern()
        
        # Determine if this should be defective
        is_defective = np.random.random() < self.defect_ratio
        
        if is_defective:
            img = self.add_defects(img)
            label = 1
        else:
            label = 0
        
        # Convert to tensor
        img = torch.FloatTensor(img).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def create_dataloaders(config):
    """Create train and validation dataloaders"""
    
    # Create datasets
    train_dataset = SyntheticSemiconductorDataset(
        num_samples=config.get('train_samples', 1000),
        image_size=config.get('image_size', 512),
        split='train'
    )
    
    val_dataset = SyntheticSemiconductorDataset(
        num_samples=config.get('val_samples', 200),
        image_size=config.get('image_size', 512),
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader