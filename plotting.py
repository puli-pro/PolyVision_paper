# --- file: plotting.py ---
"""
Visualization and plotting utilities for PolyVision framework
Generates ROC curves, calibration plots, and error analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
import os
from typing import Dict, List, Tuple, Any
from PIL import Image

plt.style.use('default')
sns.set_palette("husl")

class PolyVisionVisualizer:
    """Visualization class for PolyVision results"""
    
    def __init__(self, save_dir: str = 'figures'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_roc_curves(self, results_dict: Dict[str, Dict], save_name: str = 'roc_curves.png'):
        """
        Plot ROC curves for individual models and ensemble
        
        Args:
            results_dict: Dictionary containing results for each model
            save_name: Filename for saved plot
        """
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            fpr, tpr, _ = roc_curve(results['labels'], results['probabilities'])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.7)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_precision_recall_curves(self, results_dict: Dict[str, Dict], 
                                   save_name: str = 'pr_curves.png'):
        """Plot Precision-Recall curves"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            precision, recall, _ = precision_recall_curve(results['labels'], results['probabilities'])
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                    label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_calibration_curves(self, results_dict: Dict[str, Dict], 
                              save_name: str = 'calibration_plot.png'):
        """
        Plot calibration curves (reliability diagrams)
        
        Args:
            results_dict: Dictionary containing results for each model
            save_name: Filename for saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        # Calibration curve
        for i, (model_name, results) in enumerate(results_dict.items()):
            fraction_of_positives, mean_predicted_value = calibration_curve(
                results['labels'], results['probabilities'], n_bins=10)
            
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                    color=colors[i % len(colors)], label=model_name)
        
        # Perfect calibration line
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title('Calibration Curve (Reliability Diagram)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram of predicted probabilities
        for i, (model_name, results) in enumerate(results_dict.items()):
            ax2.hist(results['probabilities'], bins=20, alpha=0.5, 
                    color=colors[i % len(colors)], label=model_name)
        
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Predicted Probabilities', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrices(self, results_dict: Dict[str, Dict], 
                              save_name: str = 'confusion_matrices.png'):
        """Plot confusion matrices for all models"""
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            cm = confusion_matrix(results['labels'], results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No DR', 'DR'], yticklabels=['No DR', 'DR'],
                       ax=axes[i])
            axes[i].set_title(f'{model_name}', fontsize=12)
            axes[i].set_xlabel('Predicted Label')
            axes[i].set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float],
                           train_accuracies: List[float], val_accuracies: List[float],
                           save_name: str = 'training_curves.png'):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
    def error_analysis(self, results: Dict, dataloader, save_dir: str = 'error_analysis'):
        """
        Perform error analysis by identifying FP and FN cases
        
        Args:
            results: Results dictionary with labels, predictions, probabilities
            dataloader: DataLoader to get original images
            save_dir: Directory to save error analysis images
        """
        os.makedirs(save_dir, exist_ok=True)
        
        labels = results['labels']
        predictions = results['predictions']
        probabilities = results['probabilities']
        
        # Find false positives and false negatives
        fp_indices = np.where((predictions == 1) & (labels == 0))[0]
        fn_indices = np.where((predictions == 0) & (labels == 1))[0]
        
        print(f"Found {len(fp_indices)} False Positives and {len(fn_indices)} False Negatives")
        
        # Save representative cases
        self._save_error_cases(fp_indices[:5], "False_Positive", dataloader, 
                              probabilities, save_dir)
        self._save_error_cases(fn_indices[:5], "False_Negative", dataloader, 
                              probabilities, save_dir)
    
    def _save_error_cases(self, indices: np.ndarray, error_type: str, 
                         dataloader, probabilities: np.ndarray, save_dir: str):
        """Save representative error cases with annotations"""
        if len(indices) == 0:
            return
            
        # This is a simplified version - in practice, you'd need to map indices
        # back to original images in the dataset
        print(f"Saving {len(indices)} {error_type} cases to {save_dir}")
        
        # Create subdirectory
        error_dir = os.path.join(save_dir, error_type)
        os.makedirs(error_dir, exist_ok=True)
        
        # Save error case information
        with open(os.path.join(error_dir, 'error_info.txt'), 'w') as f:
            f.write(f"{error_type} Cases\n")
            f.write("=" * 20 + "\n")
            for i, idx in enumerate(indices):
                f.write(f"Case {i+1}: Index {idx}, Probability: {probabilities[idx]:.3f}\n")

