# --- file: config.py ---
"""
Configuration file containing all hyperparameters and settings for PolyVision
"""
from typing import Dict, Any

class Config:
    """Configuration class for PolyVision framework"""
    
    # Training Configuration
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 5e-3
    NUM_EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 15
    
    # Data Configuration
    IMAGE_SIZE_RESNET: int = 224
    IMAGE_SIZE_EFFICIENTNET: int = 224
    IMAGE_SIZE_VIT: int = 384  # ViT typically uses larger input size
    NUM_CLASSES: int = 2  # Binary classification: DR vs No-DR
    
    # Model Configuration
    DROPOUT_RATE: float = 0.05
    
    # Fusion Weights (for weighted averaging)
    FUSION_WEIGHTS: Dict[str, float] = {
        'resnet': 0.33,
        'efficientnet': 0.34,
        'vit': 0.33
    }
    
    # Cross-validation
    K_FOLDS: int = 5
    
    # Device
    DEVICE: str = 'cuda'
    
    # Paths
    DATA_ROOT: str = 'Diagnosis of Diabetic Retinopathy'
    RESULTS_DIR: str = 'results'
    FIGURES_DIR: str = 'figures'
    ERROR_ANALYSIS_DIR: str = 'error_analysis'
    MODEL_SAVE_DIR: str = 'saved_models'
