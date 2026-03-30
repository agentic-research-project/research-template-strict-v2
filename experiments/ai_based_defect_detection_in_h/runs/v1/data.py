import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import requests
import zipfile
from pathlib import Path
import cv2
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class MVTecDataset(Dataset):
    """MVTec AD Dataset for semiconductor defect detection proxy"""
    
    def __init__(self, root_dir: str, category: str = 'transistor', 
                 split: str = 'train', transform=None, target_size=(256, 256)):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Define data paths
        self.data_dir = self.root_dir / category / split
        self.mask_dir = self.root_dir / category / 'ground_truth' if split == 'test' else None
        
        # Load data
        self.samples = []
        self.load_samples()
    
    def load_samples(self):
        """Load all samples from the dataset"""
        if self.split == 'train':
            # Training split - only normal samples
            normal_dir = self.data_dir / 'good'
            if normal_dir.exists():
                for img_path in normal_dir.glob('*.png'):
                    self.samples.append({
                        'image_path': img_path,
                        'label': 0,  # Normal
                        'mask_path': None
                    })
        else:
            # Test split - normal and defective samples
            # Normal samples
            normal_dir = self.data_dir / 'good'
            if normal_dir.exists():
                for img_path in normal_dir.glob('*.png'):
                    self.samples.append({
                        'image_path': img_path,
                        'label': 0,  # Normal
                        'mask_path': None
                    })
            
            # Defective samples
            if self.data_dir.exists():
                for defect_dir in self.data_dir.iterdir():
                    if defect_dir.is_dir() and defect_dir.name != 'good':
                        for img_path in defect_dir.glob('*.png'):
                            mask_path = None
                            if self.mask_dir:
                                mask_path = self.mask_dir / defect_dir.name / (img_path.stem + '_mask.png')
                                if not mask_path.exists():
                                    mask_path = None
                            
                            self.samples.append({
                                'image_path': img_path,
                                'label': 1,  # Defective
                                'mask_path': mask_path
                            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to target size
        if self.target_size:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image = transforms.ToTensor()(image)
        
        # Load mask if available
        mask = None
        if sample['mask_path'] and sample['mask_path'].exists():
            mask = Image.open(sample['mask_path']).convert('L')
            if self.target_size:
                mask = mask.resize(self.target_size, Image.Resampling.NEAREST)
            mask = transforms.ToTensor()(mask)
            mask = (mask > 0.5).float()  # Binarize
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if mask is not None:
            return image, sample['label'], mask
        else:
            return image, sample['label']

class SyntheticSemiconductorDataset(Dataset):
    """Synthetic semiconductor dataset for testing"""
    
    def __init__(self, n_samples=100, image_size=(256, 256), split='train'):
        self.n_samples = n_samples
        self.image_size = image_size
        self.split = split
        
        # Generate synthetic data
        self.samples = self.generate_samples()
    
    def generate_samples(self):
        """Generate synthetic semiconductor-like images"""
        samples = []
        np.random.seed(42)
        
        for i in range(self.n_samples):
            # Create base pattern (grid-like structure)
            image = self.create_base_pattern()
            
            # Determine if this should be defective
            if self.split == 'train':
                label = 0  # Only normal samples in training
                mask = None
            else:
                label = 1 if i % 3 == 0 else 0  # ~33% defective in test
                if label == 1:
                    image, mask = self.add_defects(image)
                else:
                    mask = None
            
            samples.append({
                'image': image,
                'label': label,
                'mask': mask
            })
        
        return samples
    
    def create_base_pattern(self):
        """Create base semiconductor-like pattern"""
        h, w = self.image_size
        
        # Create grid pattern
        image = np.ones((h, w), dtype=np.float32) * 0.5
        
        # Add periodic structure
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        # Grid lines
        grid_spacing = 32
        grid_pattern = (np.sin(2 * np.pi * X / grid_spacing) > 0.5) & \
                      (np.sin(2 * np.pi * Y / grid_spacing) > 0.5)
        
        image[grid_pattern] = 0.7
        
        # Add noise
        noise = np.random.normal(0, 0.05, (h, w))
        image = np.clip(image + noise, 0, 1)
        
        return image.astype(np.float32)
    
    def add_defects(self, image):
        """Add synthetic defects to image"""
        h, w = image.shape
        mask = np.zeros((h, w), dtype=np.float32)
        defective_image = image.copy()
        
        # Random defect type
        defect_type = np.random.choice(['scratch', 'particle', 'pattern'])
        
        if defect_type == 'scratch':
            # Add scratch-like defect
            start_x = np.random.randint(0, w // 2)
            start_y = np.random.randint(0, h)
            length = np.random.randint(50, 150)
            angle = np.random.uniform(0, 2 * np.pi)
            
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            
            end_x = np.clip(end_x, 0, w - 1)
            end_y = np.clip(end_y, 0, h - 1)
            
            # Draw line
            rr, cc = self.line_coordinates(start_y, start_x, end_y, end_x, h, w)
            defective_image[rr, cc] = np.random.uniform(0.2, 0.4)
            mask[rr, cc] = 1.0
            
        elif defect_type == 'particle':
            # Add circular particle
            center_x = np.random.randint(20, w - 20)
            center_y = np.random.randint(20, h - 20)
            radius = np.random.randint(5, 15)
            
            y, x = np.ogrid[:h, :w]
            circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            
            defective_image[circle_mask] = np.random.uniform(0.8, 1.0)
            mask[circle_mask] = 1.0
            
        else:  # pattern defect
            # Add rectangular pattern defect
            x_start = np.random.randint(0, w // 2)
            y_start = np.random.randint(0, h // 2)
            width = np.random.randint(20, 40)
            height = np.random.randint(20, 40)
            
            x_end = min(x_start + width, w)
            y_end = min(y_start + height, h)
            
            defective_image[y_start:y_end, x_start:x_end] *= np.random.uniform(0.3, 0.7)
            mask[y_start:y_end, x_start:x_end] = 1.0
        
        return defective_image, mask
    
    def line_coordinates(self, r0, c0, r1, c1, h, w):
        """Generate line coordinates (simplified Bresenham's algorithm)"""
        rr, cc = [], []
        
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        
        r, c = r0, c0
        n = 1 + dr + dc
        r_inc = 1 if r1 > r0 else -1
        c_inc = 1 if c1 > c0 else -1
        error = dr - dc
        
        dr *= 2
        dc *= 2
        
        for _ in range(n):
            if 0 <= r < h and 0 <= c < w:
                rr.append(r)
                cc.append(c)
            
            if error > 0:
                r += r_inc
                error -= dc
            else:
                c += c_inc
                error += dr
        
        return np.array(rr), np.array(cc)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert to tensor
        image = torch.from_numpy(sample['image']).unsqueeze(0)  # [1, H, W]
        label = torch.tensor(sample['label'], dtype=torch.long)

        # Always return a mask so all samples have the same tuple length (collate-friendly)
        if sample['mask'] is not None:
            mask = torch.from_numpy(sample['mask']).unsqueeze(0)
        else:
            mask = torch.zeros_like(image)

        return image, label, mask

def download_mvtec_ad(data_dir: str, categories: List[str] = ['transistor']):
    """Download MVTec AD dataset"""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://www.mvtec.com/company/research/datasets/mvtec-ad"
    
    print(f"MVTec AD dataset download not implemented for automated setup.")
    print(f"Please manually download from {base_url} to {data_path}")
    print(f"Required categories: {categories}")
    
    # For now, we'll use synthetic data as fallback
    return False

def build_dataloaders(config):
    """Build data loaders for the defect detection pipeline"""
    
    # Get data configuration
    batch_size = config.get('batch_size', 16)
    num_workers = config.get('num_workers', 4)
    data_dir = config.get('data_dir', './data')
    
    # Transform pipeline
    train_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Try to use MVTec AD dataset first
    data_path = Path(data_dir)
    mvtec_path = data_path / 'mvtec_ad'
    
    if (mvtec_path / 'transistor').exists():
        print("Using MVTec AD dataset")
        # Use MVTec dataset
        train_dataset = MVTecDataset(
            root_dir=str(mvtec_path),
            category='transistor',
            split='train',
            transform=train_transform,
            target_size=(256, 256)
        )
        
        val_dataset = MVTecDataset(
            root_dir=str(mvtec_path),
            category='transistor', 
            split='test',
            transform=val_transform,
            target_size=(256, 256)
        )
        
    else:
        print("MVTec AD dataset not found. Using synthetic semiconductor dataset.")
        # Fallback to synthetic data
        train_dataset = SyntheticSemiconductorDataset(
            n_samples=200,
            image_size=(256, 256),
            split='train'
        )
        
        val_dataset = SyntheticSemiconductorDataset(
            n_samples=100,
            image_size=(256, 256),
            split='test'
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader