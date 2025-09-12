# --- file: README.md ---
# PolyVision: Multi-Model Ensemble for Diabetic Retinopathy Classification

This repository contains the complete implementation of PolyVision, a collaborative neural network ensemble for diabetic retinopathy (DR) detection as described in the research paper.

## Overview

PolyVision combines three complementary deep learning architectures:
- **ResNet50**: Local feature specialist for fine-grained pathological features
- **EfficientNet-B2**: Balanced performer with compound scaling and SE blocks
- **Vision Transformer (ViT)**: Global context expert using self-attention mechanisms

The framework implements a dual fusion mechanism:
1. **Averaged Probability Voting**: Balanced approach for general screening
2. **Maximum Confidence Voting**: High-sensitivity approach for critical cases

## Key Features

- **Modular Architecture**: Clean, object-oriented design with separate components
- **Model-Specific Augmentation**: Each model uses tailored augmentation strategies
- **Comprehensive Evaluation**: ROC curves, calibration plots, and error analysis
- **Fairness Assessment**: Post-hoc evaluation across image quality subgroups
- **Production Ready**: Complete training, evaluation, and visualization pipeline

## Installation

```bash
# Clone the repository
git clone https://github.com/puli-pro/PolyVision--Research_paper-
cd PolyVision--Research_paper-

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

Organize your diabetic retinopathy dataset as follows:
```
Diagnosis of Diabetic Retinopathy/
├── train/
│   ├── DR/
│   └── No_DR/
├── valid/
│   ├── DR/
│   └── No_DR/
└── test/
    ├── DR/
    └── No_DR/
```

## Usage

### Basic Training and Evaluation

```bash
# Run complete experiment with default settings
python main.py

# Custom configuration
python main.py --data_root /path/to/dataset --batch_size 16 --epochs 50 --lr 1e-5
```

### Key Components

1. **Configuration**: Modify `config.py` for hyperparameter tuning
2. **Individual Models**: Located in `models/` directory (resnet.py, efficientnet.py, vit.py)
3. **Ensemble Logic**: `ensemble_model.py` implements dual fusion mechanism
4. **Visualization**: `plotting.py` generates all required plots and analysis
5. **Utilities**: `utils/` contains data loading, augmentation, and metrics

### Generated Outputs

- **Models**: Saved to `saved_models/` directory
- **Figures**: ROC curves, calibration plots, confusion matrices in `figures/`
- **Results**: Comprehensive metrics saved as JSON in `results/`
- **Error Analysis**: False positive/negative cases in `error_analysis/`

## Architecture Details

### Dual Fusion Mechanism

```python
# Averaged Probability Voting (Recommended for general use)
weighted_avg = (w1 * resnet_probs + w2 * efficientnet_probs + w3 * vit_probs)

# Maximum Confidence Voting (High sensitivity for screening)
most_confident_prediction = select_highest_confidence(all_predictions)
```

### Model-Specific Augmentation

Each model uses tailored augmentation strategies to encourage complementary learning:
- **ResNet50**: ImageNet normalization + standard geometric transforms
- **EfficientNet-B2**: Dataset-specific normalization + compound augmentation
- **ViT**: Global illumination-preserving transforms for larger input size

## Performance

Based on the UWF dataset evaluation:
- **AUROC**: 0.953 ± 0.004
- **AUPRC**: 0.975 ± 0.003
- **Inference Time**: 110ms per image
- **Sensitivity**: 0.898 (averaged) / 0.912 (max confidence)
- **Specificity**: 0.925 (averaged) / 0.905 (max confidence)

## Code Structure

```
polyvision/
├── main.py                 # Main execution script
├── config.py               # Configuration and hyperparameters
├── ensemble_model.py       # PolyVision ensemble implementation
├── plotting.py             # Visualization utilities
├── models/                 # Individual model implementations
│   ├── resnet.py
│   ├── efficientnet.py
│   └── vit.py
├── utils/                  # Utility functions
│   ├── data_loader.py
│   ├── augmentation.py
│   └── metrics.py
└── requirements.txt        # Dependencies
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{polyvision2024,
  title={PolyVision: Optimising Retinal Disease Detection Through Collaborative Neural Networks},
  author={Ahmed, Sultan and Kalam, Swathi and Joshua, Eali Stephen Neal and Abdeljaber, Hikmat A. M.},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please contact the authors or open an issue on GitHub.