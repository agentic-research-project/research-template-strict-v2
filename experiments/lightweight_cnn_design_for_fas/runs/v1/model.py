import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroSE(nn.Module):
    """Micro Squeeze-and-Excitation module with high reduction ratio"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        
    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        # Global average pooling
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Squeeze and excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class EfficientMicroNet(nn.Module):
    """Lightweight CNN with micro-SE attention for parameter-efficient classification"""
    def __init__(self, config):
        super().__init__()
        num_classes = config.get('num_classes', 10)
        in_channels = config.get('in_channels', 1)
        se_reduction = config.get('se_reduction', 16)
        dropout_rate = config.get('dropout_rate', 0.3)
        
        # Feature extraction layers - optimized for <200K params
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Micro SE attention
        self.se = MicroSE(96, reduction=se_reduction)

        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(96, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, return_features=False):
        # Feature extraction
        x = self.conv1(x)  # (B, 32, 28, 28)
        x = self.conv2(x)  # (B, 64, 14, 14)
        x = self.conv3(x)  # (B, 96, 7, 7)
        x = self.conv4(x)  # (B, 96, 7, 7)

        # Attention
        x = self.se(x)     # (B, 96, 7, 7)

        # Global pooling
        features = self.global_pool(x).flatten(1)  # (B, 96)
        
        # Classification
        x = self.dropout(features)
        logits = self.classifier(x)  # (B, num_classes)
        
        if return_features:
            return logits, features
        return logits
        
    def get_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config):
    """Build EfficientMicroNet model from config"""
    model = EfficientMicroNet(config)
    
    # Log parameter count
    param_count = model.get_param_count()
    param_count_M = param_count / 1e6
    print(f"Model parameters: {param_count:,} ({param_count_M:.3f}M)")
    
    # Verify parameter budget
    budget_M = config.get('param_budget_M', 0.2)
    if param_count_M > budget_M:
        print(f"WARNING: Parameter count {param_count_M:.3f}M exceeds budget {budget_M}M")
    
    return model