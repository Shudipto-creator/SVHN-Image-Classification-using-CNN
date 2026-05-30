# SVHN Image Classification using CNN

A deep learning project that implements a Convolutional Neural Network (CNN) to classify digits from the SVHN (Street View House Numbers) dataset. This project demonstrates the application of modern CNN techniques including batch normalization, dropout regularization, and data augmentation to achieve strong performance on a real-world image classification task.

## Overview

The SVHN dataset contains digit images obtained from house numbers in Google Street View images. Unlike MNIST, SVHN presents a more challenging classification problem due to:
- Real-world lighting conditions and backgrounds
- Variations in digit appearance and perspective
- Color images (RGB) instead of grayscale
- More complex visual patterns

This project implements a CNN architecture that learns to classify these digits into 10 classes (0-9) with high accuracy.

## Features

- **Custom CNN Architecture**: 3 convolutional layers with batch normalization and max pooling
- **Regularization**: Dropout layers to prevent overfitting
- **Data Augmentation**: Random cropping and horizontal flipping for training data
- **Learning Rate Scheduling**: Adaptive learning rate adjustment based on validation performance
- **Comprehensive Evaluation**: Training curves, confusion matrix, precision/recall analysis
- **Visualization Tools**: Sample predictions, misclassification analysis

## Model Architecture

```
Input: 32x32x3 (RGB)
    |
Conv2d(3->32, 3x3, pad=1) → BatchNorm → ReLU → MaxPool2d → 16x16x32
    |
Conv2d(32->64, 3x3, pad=1) → BatchNorm → ReLU → MaxPool2d → 8x8x64
    |
Conv2d(64->128, 3x3, pad=1) → BatchNorm → ReLU → MaxPool2d → 4x4x128
    |
Flatten → 2048
    |
Linear(2048->256) → BatchNorm → ReLU → Dropout(0.5)
    |
Linear(256->128) → BatchNorm → ReLU → Dropout(0.5)
    |
Linear(128->10) → logits (0-9)
```

### Key Components

- **Convolutional Layers**: Extract hierarchical features from images
- **Batch Normalization**: Stabilizes training and accelerates convergence
- **Max Pooling**: Reduces spatial dimensions while preserving important features
- **Dropout**: Regularization technique to prevent overfitting
- **Fully Connected Layers**: Final classification based on extracted features

## Setup

### Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- numpy
- scikit-learn

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd SVHN_CNN
```

2. Install required packages:
```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

3. Download the SVHN dataset:

**Option A: Automatic Download (Recommended)**
The dataset will be automatically downloaded on first run when you execute the training script. Simply run:
```bash
python svhn_cnn.py
```
The script will download the dataset to the `data/` directory automatically.

**Option B: Manual Download**
If you prefer to download the dataset manually, get it from the [official SVHN website](http://ufldl.stanford.edu/housenumbers/):
- Download `train_32x32.mat` (Training set)
- Download `test_32x32.mat` (Test set)
- Place both files in the `data/` directory in your project folder

The dataset files are quite large (approximately 1GB combined), so automatic download is recommended for convenience.

## Usage

### Training the Model

Run the main training script:
```bash
python svhn_cnn.py
```

This will:
- Load and preprocess the SVHN dataset
- Train the CNN for 15 epochs with data augmentation
- Save the best model based on test accuracy
- Generate training curves and sample predictions
- Output results to `outputs_topic2/` directory

### Generating Additional Analysis Plots

After training, generate detailed evaluation plots:
```bash
python generate_extra_plots.py
```

This creates:
- Full confusion matrix (10x10)
- Per-class precision and recall bar chart
- Grid of misclassified test images

Results are saved to `extra_plots/` directory.

## Hyperparameters

- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Epochs**: 15
- **Optimizer**: Adam with weight decay (1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2)
- **Dropout Rate**: 0.5

## Results

The model achieves competitive accuracy on the SVHN test set. Training progress is tracked through:
- Loss curves (training and test)
- Accuracy curves (training and test)
- Best model checkpointing
- Final evaluation metrics

### Output Files

- `outputs_topic2/models/best_svhn_cnn.pth`: Best model checkpoint
- `outputs_topic2/models/final_svhn_cnn.pth`: Final model after training
- `outputs_topic2/plots/training_curves.png`: Loss and accuracy over epochs
- `outputs_topic2/plots/sample_predictions.png`: Sample test predictions
- `extra_plots/confusion_matrix_full.png`: Detailed confusion matrix
- `extra_plots/per_class_precision_recall.png`: Per-class performance metrics
- `extra_plots/misclassifications.png`: Examples of misclassified images

## Project Structure

```
SVHN_CNN/
├── svhn_cnn.py                 # Main training script
├── generate_extra_plots.py     # Additional evaluation visualizations
├── data/                       # SVHN dataset directory
├── outputs_topic2/             # Training outputs
│   ├── models/                 # Saved model checkpoints
│   └── plots/                  # Training curves and predictions
├── extra_plots/                # Additional analysis plots
└── README.md                   # This file
```

## Technical Details

### Data Preprocessing

- **Training**: Random crop (32x32 with padding=4), random horizontal flip (p=0.5), normalization
- **Testing**: Simple normalization (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

### Training Strategy

- Cross-entropy loss for multi-class classification
- Adam optimizer with L2 regularization
- Learning rate reduction on validation accuracy plateau
- Model checkpointing based on best test accuracy

### Evaluation Metrics

- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix analysis
- Visual inspection of predictions and misclassifications

## Notes

- The model uses `num_workers=0` in DataLoader to avoid multiprocessing issues on Windows
- GPU acceleration is automatically used if CUDA is available
- The architecture is designed to balance performance with computational efficiency

## Future Improvements

Potential enhancements for this project:
- Implement more advanced architectures (ResNet, DenseNet)
- Add more aggressive data augmentation techniques
- Experiment with different optimizers and learning rate schedules
- Implement ensemble methods for improved accuracy
- Add support for transfer learning from pre-trained models

## License

This project is provided for educational purposes.

## Acknowledgments

- SVHN Dataset: Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng. "Reading Digits in Natural Images with Unsupervised Feature Learning." NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011.
- PyTorch team for the deep learning framework
- Scikit-learn for evaluation metrics utilities
