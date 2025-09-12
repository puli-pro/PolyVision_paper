# --- file: models/vit.py ---
"""
Vision Transformer (ViT) implementation for PolyVision framework
Global context expert using self-attention mechanisms
"""
import torch
import torch.nn as nn
try:
    from pytorch_pretrained_vit import ViT
except ImportError:
    # Fallback to timm
    import timm

class ViTDRClassifier(nn.Module):
    """Vision Transformer based classifier for diabetic retinopathy detection"""
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.05):
        """
        Initialize ViT classifier
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(ViTDRClassifier, self).__init__()
        
        try:
            # Try using pytorch_pretrained_vit
            self.backbone = ViT('B_16_imagenet1k', pretrained=True)
            self.image_size = self.backbone.image_size
            
            # Replace classifier head
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, num_classes)
            )
            
        except:
            # Fallback to timm
            self.backbone = timm.create_model('vit_base_patch16_224', 
                                            pretrained=True, 
                                            num_classes=num_classes)
            self.image_size = 224
            
            # Add dropout to classifier
            if hasattr(self.backbone, 'head'):
                in_features = self.backbone.head.in_features
                self.backbone.head = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before final classification layer"""
        if hasattr(self.backbone, 'transformer'):
            # pytorch_pretrained_vit
            x = self.backbone.conv(x)
            x = x.flatten(2).transpose(1, 2)
            x = torch.cat([self.backbone.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = self.backbone.transformer(x + self.backbone.positional_embedding)
            return x[:, 0]  # Return CLS token
        else:
            # timm fallback
            return self.backbone.forward_features(x)