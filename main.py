# --- file: main.py ---
"""
Main execution script for PolyVision framework
Handles training, evaluation, and visualization
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import argparse
from typing import Dict, List
import json

from config import Config
from utils.data_loader import DataLoaderFactory
from utils.augmentation import ModelSpecificAugmentation
from utils.metrics import MetricsCalculator, CrossValidationResults
from ensemble_model import PolyVisionEnsemble, PolyVisionTrainer
from plotting import PolyVisionVisualizer

class PolyVisionExperiment:
    """Main experiment class for PolyVision framework"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        
        # Initialize components
        self.data_factory = DataLoaderFactory(config.DATA_ROOT, config.BATCH_SIZE)
        self.augmentation = ModelSpecificAugmentation()
        self.metrics_calc = MetricsCalculator()
        self.visualizer = PolyVisionVisualizer(config.FIGURES_DIR)
        
        print(f"Initialized PolyVision experiment on device: {self.device}")
        
    def setup_data_loaders(self) -> Dict:
        """Setup data loaders with model-specific augmentations"""
        transforms_dict = {
            'resnet': self.augmentation.get_resnet_transforms(self.config.IMAGE_SIZE_RESNET),
            'efficientnet': self.augmentation.get_efficientnet_transforms(self.config.IMAGE_SIZE_EFFICIENTNET),
            'vit': self.augmentation.get_vit_transforms(self.config.IMAGE_SIZE_VIT)
        }
        
        return self.data_factory.create_loaders(transforms_dict)
    
    def train_single_model(self, model: nn.Module, train_loader, val_loader, 
                          model_name: str) -> Dict:
        """Train a single model"""
        optimizer = optim.Adam(model.parameters(), 
                             lr=self.config.LEARNING_RATE,
                             weight_decay=self.config.WEIGHT_DECAY)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.NUM_EPOCHS)
        criterion = nn.CrossEntropyLoss()
        
        trainer = PolyVisionTrainer(model, self.device)
        
        # Training history
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        print(f"\nTraining {model_name}...")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training
            train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_results = trainer.evaluate(val_loader, criterion)
            val_loss = val_results['loss']
            val_acc = val_results['accuracy']
            
            # Learning rate scheduling
            scheduler.step()
            
            # Track metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 
                          os.path.join(self.config.MODEL_SAVE_DIR, f'best_{model_name}.pth'))
            else:
                patience_counter += 1
                
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
            # Early stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping for ensemble at epoch {epoch+1}")
                break
        
        # Load best model
        ensemble_model.load_state_dict(torch.load(os.path.join(self.config.MODEL_SAVE_DIR, 'best_ensemble.pth')))
        
        return {
            'model': ensemble_model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy,
            'data_loaders': data_loaders
        }
    
    def train_ensemble(self) -> Dict:
        """Train the complete PolyVision ensemble"""
        print("Setting up data loaders...")
        data_loaders = self.setup_data_loaders()
        
        # Initialize ensemble model
        ensemble_model = PolyVisionEnsemble(
            num_classes=self.config.NUM_CLASSES,
            dropout_rate=self.config.DROPOUT_RATE,
            fusion_weights=self.config.FUSION_WEIGHTS
        )
        
        # Setup training
        optimizer = optim.Adam(ensemble_model.parameters(),
                             lr=self.config.LEARNING_RATE,
                             weight_decay=self.config.WEIGHT_DECAY)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.NUM_EPOCHS)
        criterion = nn.CrossEntropyLoss()
        
        trainer = PolyVisionTrainer(ensemble_model, self.device)
        
        # Training history
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        print("Training PolyVision ensemble...")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training
            train_loss, train_acc = trainer.train_epoch(
                data_loaders['train'], optimizer, criterion, 'averaged')
            
            # Validation
            val_results = trainer.evaluate(
                data_loaders['valid'], criterion, 'averaged')
            val_loss = val_results['loss']
            val_acc = val_results['accuracy']
            
            # Learning rate scheduling
            scheduler.step()
            
            # Track metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                # Save best model
                torch.save(ensemble_model.state_dict(),
                          os.path.join(self.config.MODEL_SAVE_DIR, 'best_ensemble.pth'))
            else:
                patience_counter += 1
                
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
                
            # Early stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping for {model_name} at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(self.config.MODEL_SAVE_DIR, f'best_{model_name}.pth')))
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_accuracy
        }
    
    def train_ensemble(self) -> Dict:
        """Train the complete PolyVision ensemble"""
        print("Setting up data loaders...")
        data_loaders = self.setup_data_loaders()
        
        # Initialize ensemble model
        ensemble_model = PolyVisionEnsemble(
            num_classes=self.config.NUM_CLASSES,
            dropout_rate=self.config.DROPOUT_RATE,
            fusion_weights=self.config.FUSION_WEIGHTS
        )
        
        # Setup training
        optimizer = optim.Adam(ensemble_model.parameters(),
                             lr=self.config.LEARNING_RATE,
                             weight_decay=self.config.WEIGHT_DECAY)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.NUM_EPOCHS)
        criterion = nn.CrossEntropyLoss()
        
        trainer = PolyVisionTrainer(ensemble_model, self.device)
        
        # Training history
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        
        best_val_accuracy = 0.0
        patience_counter = 0
        
        print("Training PolyVision ensemble...")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training
            train_loss, train_acc = trainer.train_epoch(
                data_loaders['train'], optimizer, criterion, 'averaged')
            
            # Validation
            val_results = trainer.evaluate(
                data_loaders['valid'], criterion, 'averaged')
            val_loss = val_results['loss']
            val_acc = val_results['accuracy']
            
            # Learning rate scheduling
            scheduler.step()
            
            # Track metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Early stopping
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                # Save best model
                torch.save(ensemble_model.state_dict(),
                          os.path.join(self.config.MODEL_SAVE_DIR, 'best_ensemble.pth'))
            else:
                patience_counter += 1
                
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.config.NUM_EPOCHS}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    
    
    def evaluate_model(self, model: PolyVisionEnsemble, test_loader, 
                      fusion_strategies: List[str] = ['averaged', 'max_conf']) -> Dict:
        """Comprehensive evaluation of the ensemble model"""
        trainer = PolyVisionTrainer(model, self.device)
        criterion = nn.CrossEntropyLoss()
        
        results = {}
        
        for strategy in fusion_strategies:
            print(f"Evaluating with {strategy} fusion strategy...")
            eval_results = trainer.evaluate(test_loader, criterion, strategy)
            
            # Calculate additional metrics
            auroc, auprc = self.metrics_calc.calculate_auroc_auprc(
                eval_results['labels'], eval_results['probabilities'])
            
            sensitivity, specificity = self.metrics_calc.calculate_sensitivity_specificity(
                eval_results['labels'], eval_results['predictions'])
            
            ece = self.metrics_calc.expected_calibration_error(
                eval_results['labels'], eval_results['probabilities'])
            
            results[f'ensemble_{strategy}'] = {
                'labels': eval_results['labels'],
                'probabilities': eval_results['probabilities'],
                'predictions': eval_results['predictions'],
                'accuracy': eval_results['accuracy'],
                'auroc': auroc,
                'auprc': auprc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ece': ece
            }
            
            print(f"{strategy} - Accuracy: {eval_results['accuracy']:.4f}, "
                  f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
        
        return results
    
    def evaluate_individual_models(self, ensemble_model: PolyVisionEnsemble, 
                                 test_loader) -> Dict:
        """Evaluate individual models within the ensemble"""
        ensemble_model.eval()
        
        all_individual_results = {
            'resnet': {'labels': [], 'probabilities': [], 'predictions': []},
            'efficientnet': {'labels': [], 'probabilities': [], 'predictions': []},
            'vit': {'labels': [], 'probabilities': [], 'predictions': []}
        }
        
        with torch.no_grad():
            for batch in test_loader:
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                labels = batch['label']
                
                # Get individual model predictions
                individual_preds = ensemble_model.get_individual_predictions(batch)
                
                for model_name in ['resnet', 'efficientnet', 'vit']:
                    probs = individual_preds[f'{model_name}_probs']
                    predictions = torch.argmax(probs, dim=1)
                    
                    all_individual_results[model_name]['labels'].extend(labels.cpu().numpy())
                    all_individual_results[model_name]['probabilities'].extend(
                        probs[:, 1].cpu().numpy())
                    all_individual_results[model_name]['predictions'].extend(
                        predictions.cpu().numpy())
        
        # Calculate metrics for each individual model
        individual_results = {}
        for model_name, data in all_individual_results.items():
            labels = np.array(data['labels'])
            probabilities = np.array(data['probabilities'])
            predictions = np.array(data['predictions'])
            
            accuracy = np.mean(predictions == labels)
            auroc, auprc = self.metrics_calc.calculate_auroc_auprc(labels, probabilities)
            sensitivity, specificity = self.metrics_calc.calculate_sensitivity_specificity(
                labels, predictions)
            ece = self.metrics_calc.expected_calibration_error(labels, probabilities)
            
            individual_results[model_name] = {
                'labels': labels,
                'probabilities': probabilities,
                'predictions': predictions,
                'accuracy': accuracy,
                'auroc': auroc,
                'auprc': auprc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ece': ece
            }
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, "
                  f"AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
        
        return individual_results
    
    def generate_visualizations(self, ensemble_results: Dict, individual_results: Dict):
        """Generate all required visualizations"""
        print("Generating visualizations...")
        
        # Combine all results for plotting
        all_results = {**individual_results, **ensemble_results}
        
        # ROC Curves
        self.visualizer.plot_roc_curves(all_results, 'roc_curves.png')
        
        # Precision-Recall Curves
        self.visualizer.plot_precision_recall_curves(all_results, 'pr_curves.png')
        
        # Calibration plots
        self.visualizer.plot_calibration_curves(all_results, 'calibration_plot.png')
        
        # Confusion matrices
        self.visualizer.plot_confusion_matrices(all_results, 'confusion_matrices.png')
        
        print("Visualizations saved to", self.config.FIGURES_DIR)
    
    def save_results(self, results: Dict, filename: str = 'experiment_results.json'):
        """Save experiment results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {}
            for metric_name, metric_value in model_results.items():
                if isinstance(metric_value, np.ndarray):
                    serializable_results[model_name][metric_name] = metric_value.tolist()
                else:
                    serializable_results[model_name][metric_name] = metric_value
        
        results_path = os.path.join(self.config.RESULTS_DIR, filename)
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_path}")
    
    def run_complete_experiment(self):
        """Run the complete PolyVision experiment"""
        print("=" * 50)
        print("POLYVISION EXPERIMENT")
        print("=" * 50)
        
        # Train ensemble
        training_results = self.train_ensemble()
        ensemble_model = training_results['model']
        test_loader = training_results['data_loaders']['test']
        
        # Plot training curves
        self.visualizer.plot_training_curves(
            training_results['train_losses'],
            training_results['val_losses'], 
            training_results['train_accuracies'],
            training_results['val_accuracies'],
            'ensemble_training_curves.png'
        )
        
        # Evaluate ensemble with both fusion strategies
        print("\nEvaluating ensemble model...")
        ensemble_results = self.evaluate_model(ensemble_model, test_loader)
        
        # Evaluate individual models
        print("\nEvaluating individual models...")
        individual_results = self.evaluate_individual_models(ensemble_model, test_loader)
        
        # Generate visualizations
        self.generate_visualizations(ensemble_results, individual_results)
        
        # Perform error analysis
        print("\nPerforming error analysis...")
        self.visualizer.error_analysis(
            ensemble_results['ensemble_averaged'], 
            test_loader, 
            self.config.ERROR_ANALYSIS_DIR
        )
        
        # Save all results
        all_results = {**individual_results, **ensemble_results}
        self.save_results(all_results)
        
        # Print final summary
        self.print_final_summary(all_results)
        
        print("\nExperiment completed successfully!")
        
    def print_final_summary(self, results: Dict):
        """Print final experimental results summary"""
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"{'Model':<20} {'Accuracy':<10} {'AUROC':<8} {'AUPRC':<8} {'Sensitivity':<12} {'Specificity':<12}")
        print("-" * 70)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<20} {metrics['accuracy']:<10.4f} {metrics['auroc']:<8.4f} "
                  f"{metrics['auprc']:<8.4f} {metrics['sensitivity']:<12.4f} {metrics['specificity']:<12.4f}")
        
        print("\nBest performing model (by AUROC):")
        best_model = max(results.items(), key=lambda x: x[1]['auroc'])
        print(f"{best_model[0]}: AUROC = {best_model[1]['auroc']:.4f}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='PolyVision: Multi-model Ensemble for DR Classification')
    parser.add_argument('--data_root', type=str, default='Diagnosis of Diabetic Retinopathy',
                      help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = Config()
    config.DATA_ROOT = args.data_root
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.DEVICE = args.device
    
    # Run experiment
    experiment = PolyVisionExperiment(config)
    experiment.run_complete_experiment()

if __name__ == "__main__":
    main()


