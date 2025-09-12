# --- file: utils/augmentation.py ---
"""
Data augmentation strategies for PolyVision framework
Each model uses specific augmentation to encourage complementary learning
"""
import torch
import torchvision.transforms as transforms
from typing import Dict, Any

class ModelSpecificAugmentation:
    """Model-specific augmentation strategies"""
    
    @staticmethod
    def get_resnet_transforms(image_size: int = 224) -> transforms.Compose:
        """ResNet50 augmentation with ImageNet normalization"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_efficientnet_transforms(image_size: int = 224) -> transforms.Compose:
        """EfficientNet-B2 augmentation with dataset-specific normalization"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                               std=[0.5, 0.5, 0.5])
        ])
    
    @staticmethod
    def get_vit_transforms(image_size: int = 384) -> transforms.Compose:
        """ViT augmentation with global illumination pattern preservation"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_test_transforms(image_size: int = 224) -> transforms.Compose:
        """Test-time transforms (no augmentation)"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
