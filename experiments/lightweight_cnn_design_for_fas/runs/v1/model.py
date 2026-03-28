import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Use 1x1 convs instead of Linear to keep everything convolutional and lightweight
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class DSCNNSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, se_reduction=4):
        super(DSCNNSEBlock, self).__init__()
        
        # Depthwise separable convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE module
        self.se = SEModule(out_channels, se_reduction)
        
        # Residual connection projection if needed
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        # Depthwise separable convolution
        out = F.relu(self.bn1(self.depthwise(x)), inplace=True)
        out = F.relu(self.bn2(self.pointwise(out)), inplace=True)
        
        # SE module
        out = self.se(out)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        out = out + residual
        return out

class NonUniformDSCNNSE(nn.Module):
    def __init__(self, config):
        super(NonUniformDSCNNSE, self).__init__()
        
        num_classes = config.get('num_classes', 10)
        
        # Stage 1: Initial feature extraction (wide + shallow)
        self.stage1_conv = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.stage1_block = DSCNNSEBlock(48, 48, se_reduction=8)
        self.stage1_pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        
        # Stage 2: Mid-level feature extraction (medium depth)
        self.stage2_block1 = DSCNNSEBlock(48, 96, se_reduction=4)
        self.stage2_block2 = DSCNNSEBlock(96, 96, se_reduction=4)
        self.stage2_pool = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        
        # Stage 3: High-level feature extraction (narrow + deep)
        self.stage3_block1 = DSCNNSEBlock(96, 128, se_reduction=4)
        self.stage3_block2 = DSCNNSEBlock(128, 128, se_reduction=4)

        # Classification head: Global Average Pooling -> FC
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        
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
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stage 1
        x = self.stage1_conv(x)
        x = self.stage1_block(x)
        x = self.stage1_pool(x)

        # Stage 2
        x = self.stage2_block1(x)
        x = self.stage2_block2(x)
        x = self.stage2_pool(x)

        # Stage 3
        x = self.stage3_block1(x)
        x = self.stage3_block2(x)

        # Global average pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def build_model(config):
    return NonUniformDSCNNSE(config)