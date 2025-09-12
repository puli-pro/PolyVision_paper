# --- file: ensemble_model.py ---
"""
PolyVision ensemble model implementing dual fusion mechanism
Combines ResNet50, EfficientNet-B2, and ViT for robust DR classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import numpy as np

from models.resnet import ResNetDRClassifier
from models.efficientnet import EfficientNetDRClassifier
from models.vit import ViTDRClassifier

class PolyVisionEnsemble(nn.Module):
    """
    PolyVision ensemble implementing dual fusion mechanism:
    1. Averaged Probability Voting
    2. Maximum Confidence Voting
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.05,
                 fusion_weights: Dict[str, float] = None):
        """
        Initialize PolyVision ensemble
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for individual models
            fusion_weights: Weights for averaged probability voting
        """
        super(PolyVisionEnsemble, self).__init__()
        
        # Initialize individual models
        self.resnet = ResNetDRClassifier(num_classes, dropout_rate)
        self.efficientnet = EfficientNetDRClassifier(num_classes, dropout_rate)
        self.vit = ViTDRClassifier(num_classes, dropout_rate)
        
        # Fusion weights
        if fusion_weights is None:
            self.fusion_weights = {'resnet': 0.33, 'efficientnet': 0.34, 'vit': 0.33}
        else:
            self.fusion_weights = fusion_weights
            
        self.num_classes = num_classes
        
    def forward(self, batch: Dict[str, torch.Tensor], 
                return_individual: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble
        
        Args:
            batch: Dictionary containing images for each model
            return_individual: Whether to return individual model predictions
            
        Returns:
            Dictionary containing ensemble predictions and optionally individual predictions
        """
        # Get predictions from individual models
        resnet_logits = self.resnet(batch['resnet_image'])
        efficientnet_logits = self.efficientnet(batch['efficientnet_image'])
        vit_logits = self.vit(batch['vit_image'])
        
        # Convert to probabilities
        resnet_probs = F.softmax(resnet_logits, dim=1)
        efficientnet_probs = F.softmax(efficientnet_logits, dim=1)
        vit_probs = F.softmax(vit_logits, dim=1)
        
        # Dual fusion mechanism
        averaged_probs = self._averaged_probability_voting(
            resnet_probs, efficientnet_probs, vit_probs)
        
        max_conf_probs = self._maximum_confidence_voting(
            resnet_probs, efficientnet_probs, vit_probs)
        
        result = {
            'averaged_probs': averaged_probs,
            'max_conf_probs': max_conf_probs,
            'averaged_logits': torch.log(averaged_probs + 1e-8),  # For loss calculation
            'max_conf_logits': torch.log(max_conf_probs + 1e-8)
        }
        
        if return_individual:
            result.update({
                'resnet_probs': resnet_probs,
                'efficientnet_probs': efficientnet_probs,
                'vit_probs': vit_probs,
                'resnet_logits': resnet_logits,
                'efficientnet_logits': efficientnet_logits,
                'vit_logits': vit_logits
            })
            
        return result
    
    def _averaged_probability_voting(self, resnet_probs: torch.Tensor,
                                   efficientnet_probs: torch.Tensor,
                                   vit_probs: torch.Tensor) -> torch.Tensor:
        """
        Averaged probability voting fusion strategy
        Provides balanced and robust prediction
        """
        weighted_avg = (
            self.fusion_weights['resnet'] * resnet_probs +
            self.fusion_weights['efficientnet'] * efficientnet_probs +
            self.fusion_weights['vit'] * vit_probs
        )
        return weighted_avg
    
    def _maximum_confidence_voting(self, resnet_probs: torch.Tensor,
                                 efficientnet_probs: torch.Tensor,
                                 vit_probs: torch.Tensor) -> torch.Tensor:
        """
        Maximum confidence voting fusion strategy
        Selects prediction from most confident model
        """
        batch_size = resnet_probs.shape[0]
        result_probs = torch.zeros_like(resnet_probs)
        
        # Stack all probabilities
        all_probs = torch.stack([resnet_probs, efficientnet_probs, vit_probs], dim=0)
        
        # Find maximum confidence for each sample
        max_confidences = torch.max(all_probs, dim=2)[0]  # Shape: [3, batch_size]
        most_confident_model = torch.argmax(max_confidences, dim=0)  # Shape: [batch_size]
        
        # Select probabilities from most confident model for each sample
        for i in range(batch_size):
            model_idx = most_confident_model[i].item()
            result_probs[i] = all_probs[model_idx, i]
            
        return result_probs
    
    def get_individual_predictions(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get predictions from individual models"""
        with torch.no_grad():
            resnet_logits = self.resnet(batch['resnet_image'])
            efficientnet_logits = self.efficientnet(batch['efficientnet_image'])
            vit_logits = self.vit(batch['vit_image'])
            
        return {
            'resnet_probs': F.softmax(resnet_logits, dim=1),
            'efficientnet_probs': F.softmax(efficientnet_logits, dim=1),
            'vit_probs': F.softmax(vit_logits, dim=1)
        }

class PolyVisionTrainer:
    """Trainer class for PolyVision ensemble"""
    
    def __init__(self, model: PolyVisionEnsemble, device: torch.device):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, dataloader, optimizer, criterion, fusion_strategy: str = 'averaged'):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in dataloader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            labels = batch['label']
            
            # Forward pass
            outputs = self.model(batch)
            
            # Select fusion strategy
            if fusion_strategy == 'averaged':
                logits = outputs['averaged_logits']
                probs = outputs['averaged_probs']
            else:  # max_conf
                logits = outputs['max_conf_logits']
                probs = outputs['max_conf_probs']
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(probs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader, criterion, fusion_strategy: str = 'averaged'):
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                labels = batch['label']
                
                # Forward pass
                outputs = self.model(batch)
                
                # Select fusion strategy
                if fusion_strategy == 'averaged':
                    logits = outputs['averaged_logits']
                    probs = outputs['averaged_probs']
                else:  # max_conf
                    logits = outputs['max_conf_logits']
                    probs = outputs['max_conf_probs']
                
                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # Store predictions
                predictions = torch.argmax(probs, dim=1)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Positive class probabilities
                all_predictions.extend(predictions.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'labels': np.array(all_labels),
            'probabilities': np.array(all_probs),
            'predictions': np.array(all_predictions)
        }