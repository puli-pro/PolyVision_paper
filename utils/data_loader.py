# --- file: utils/data_loader.py ---
"""
Data loading utilities for PolyVision framework
Handles multi-transform dataset loading for different models
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image

class MultiTransformDataset(Dataset):
    """Dataset that applies different transforms for different models"""
    
    def __init__(self, dataset: ImageFolder, transforms_dict: Dict[str, Any]):
        """
        Args:
            dataset: Base ImageFolder dataset
            transforms_dict: Dictionary of transforms for each model
        """
        self.dataset = dataset
        self.transforms_dict = transforms_dict
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns transformed images for each model"""
        image, label = self.dataset[idx]
        
        result = {'label': torch.tensor(label, dtype=torch.long)}
        
        for model_name, transform in self.transforms_dict.items():
            result[f'{model_name}_image'] = transform(image)
            
        return result

class DataLoaderFactory:
    """Factory class for creating data loaders"""
    
    def __init__(self, data_root: str, batch_size: int = 32):
        self.data_root = data_root
        self.batch_size = batch_size
        
    def create_loaders(self, transforms_dict: Dict[str, Any], 
                      fold_indices: Dict[str, List[int]] = None) -> Dict[str, DataLoader]:
        """
        Create data loaders for train/val/test splits
        
        Args:
            transforms_dict: Dictionary of transforms for each model
            fold_indices: Optional dictionary containing train/val/test indices
            
        Returns:
            Dictionary containing DataLoader objects
        """
        # Load base datasets
        train_dataset = ImageFolder(root=os.path.join(self.data_root, 'train'))
        test_dataset = ImageFolder(root=os.path.join(self.data_root, 'test'))
        valid_dataset = ImageFolder(root=os.path.join(self.data_root, 'valid'))
        
        # Create multi-transform datasets
        train_multi = MultiTransformDataset(train_dataset, transforms_dict)
        test_multi = MultiTransformDataset(test_dataset, transforms_dict)
        valid_multi = MultiTransformDataset(valid_dataset, transforms_dict)
        
        # Create data loaders
        loaders = {
            'train': DataLoader(train_multi, batch_size=self.batch_size, 
                              shuffle=True, num_workers=4),
            'valid': DataLoader(valid_multi, batch_size=self.batch_size, 
                              shuffle=False, num_workers=4),
            'test': DataLoader(test_multi, batch_size=self.batch_size, 
                             shuffle=False, num_workers=4)
        }
        
        return loaders