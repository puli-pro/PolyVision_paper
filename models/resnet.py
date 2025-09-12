# --- file: models/resnet.py ---
"""
ResNet50 implementation for PolyVision framework
Local feature specialist for fine-grained retinal features
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple

class ResNetDRClassifier(nn.Module):
    """ResNet50-based classifier for diabetic retinopathy detection"""
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.05):
        """
        Initialize ResNet50 classifier
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(ResNetDRClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze backbone parameters (transfer learning)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Get feature dimension
        feature_dim = self.backbone.fc.in_features
        
        # Replace classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Unfreeze final classifier for training
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification layer"""
        # Forward through all layers except final classifier
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x