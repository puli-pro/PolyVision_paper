# --- file: models/efficientnet.py ---
"""
EfficientNet-B2 implementation for PolyVision framework
Balanced performer with compound scaling
"""
import torch
import torch.nn as nn
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    # Fallback to timm if efficientnet_pytorch is not available
    import timm

class EfficientNetDRClassifier(nn.Module):
    """EfficientNet-B2 based classifier for diabetic retinopathy detection"""
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.05):
        """
        Initialize EfficientNet-B2 classifier
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(EfficientNetDRClassifier, self).__init__()
        
        try:
            # Try using efficientnet_pytorch
            self.backbone = EfficientNet.from_pretrained('efficientnet-b2', 
                                                       num_classes=num_classes)
        except:
            # Fallback to timm
            self.backbone = timm.create_model('efficientnet_b2', 
                                            pretrained=True, 
                                            num_classes=num_classes)
            
        # Add dropout to classifier if not present
        if hasattr(self.backbone, '_dropout'):
            self.backbone._dropout = nn.Dropout(dropout_rate)
        elif hasattr(self.backbone, 'classifier'):
            # For timm models
            if isinstance(self.backbone.classifier, nn.Linear):
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification layer"""
        if hasattr(self.backbone, 'extract_features'):
            # efficientnet_pytorch
            features = self.backbone.extract_features(x)
            features = self.backbone._avg_pooling(features)
            features = features.flatten(start_dim=1)
            return features
        else:
            # timm fallback
            features = self.backbone.forward_features(x)
            if features.dim() == 4:  # [B, C, H, W]
                features = torch.mean(features, dim=[2, 3])  # Global average pooling
            return features