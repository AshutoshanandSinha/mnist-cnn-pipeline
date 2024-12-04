# MNIST Classification Model

This project implements a CNN model for MNIST digit classification, achieving 99.40% test accuracy.

## Model Architecture

The model uses a modern CNN architecture with the following key features:

- **Input Layer**: Accepts 28x28 grayscale images
- **Feature Extraction Blocks**:
  - Initial features (8 channels) → Deep features (12→16→20 channels)
  - No padding in early layers for aggressive feature extraction
  - BatchNorm and ReLU activation after each convolution
- **Spatial Reduction**: MaxPool2d reduces spatial dimensions by 2x
- **Channel Compression**: 1x1 convolution reduces channels from 20→12
- **Feature Refinement**: Three blocks with padding maintain spatial dimensions
- **Classification**: Global Average Pooling → 1x1 Conv for final predictions

Total Parameters: 12,534

## Training Details

- **Dataset**: MNIST (60,000 training, 10,000 test images)
- **Preprocessing**:
  - Normalization (calculated from dataset)
  - Data Augmentation:
    - Random rotation (±18°)
    - Random affine transforms (translation ±0.1, scale 0.8-1.2)

- **Training Configuration**:
  - Optimizer: Adam (lr=0.001)
  - Scheduler: OneCycleLR (max_lr=0.01)
  - Batch Size: 124
  - Loss Function: CrossEntropyLoss
  - Gradient Clipping: 1.0
  - Early Stopping: 5 epochs patience

## Results

The model achieved:
- Test Accuracy: 99.40%
- Training Time: 14 epochs
- Early Stopping: Triggered upon reaching target accuracy

## Training Progress

Epoch 1 - Loss: 1.9788, Accuracy: 42.24%, Test: 75.45%
Epoch 5 - Loss: 0.5840, Accuracy: 98.39%, Test: 98.80%
Epoch 10 - Loss: 0.5509, Accuracy: 99.04%, Test: 99.18%
Epoch 14 - Loss: 0.5417, Accuracy: 99.23%, Test: 99.40%

## Key Features

1. **Efficient Architecture**:
   - Progressive channel expansion
   - Strategic use of padding/no-padding
   - Channel compression to reduce parameters

2. **Training Optimizations**:
   - Learning rate scheduling
   - Gradient clipping
   - Label smoothing
   - Data augmentation

3. **Monitoring & Control**:
   - TensorBoard integration
   - Early stopping
   - Best model checkpointing

## Requirements

- PyTorch
- torchvision
- tqdm
- tensorboard

## Usage

1. Train the model:

```bash
python src/train.py
```

2. Monitor training:

```bash
tensorboard --logdir=runs/mnist_training
```

## Model Checkpoints

The training process saves:
- `best_model.pth`: Best performing model weights
- `model_checkpoint.pth`: Complete training state including optimizer

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Project Structure
├── src/
│ ├── model.py # Model architecture
│ └── train.py # Training pipeline
├── .github/
│ ├── workflows/ # GitHub Actions
│ └── scripts/ # Test scripts
└── requirements.txt # Dependencies

## Setup and Installation
bash
Clone the repository
git clone https://github.com/AshutoshanandSinha/mnist-cnn-pipeline.git
Install dependencies
pip install -r requirements.txt

## Training

bash:assignment5/README.Md
python src/train.py

## Code Quality

[![Code Checks](https://github.com/AshutoshanandSinha/mnist-cnn-pipeline/actions/workflows/code_checks.yml/badge.svg)](https://github.com/AshutoshanandSinha/mnist-cnn-pipeline/actions/workflows/code_checks.yml)

The codebase is checked for:
- PEP8 compliance (flake8)
- Model architecture requirements
- Parameter count validation
- Use of BatchNorm and GAP

## Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0
- tqdm >= 4.65.0
- flake8 >= 3.9.0

## Links

- [Test Logs](https://github.com/AshutoshanandSinha/mnist-cnn-pipeline/actions)
- [GitHub Actions](https://github.com/AshutoshanandSinha/mnist-cnn-pipeline/actions/workflows/code_checks.yml)
- [Model Implementation](https://github.com/AshutoshanandSinha/mnist-cnn-pipeline/blob/main/src/model.py)

## License

MIT
