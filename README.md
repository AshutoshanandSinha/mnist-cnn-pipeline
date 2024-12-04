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
- Test Accuracy: 99.41%
- Training Time: 10 epochs
- Early Stopping: Triggered upon reaching target accuracy

### Model Architecture Details
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
       BatchNorm2d-2            [-1, 8, 26, 26]              16
              ReLU-3            [-1, 8, 26, 26]               0
            Conv2d-4           [-1, 12, 24, 24]             864
       BatchNorm2d-5           [-1, 12, 24, 24]              24
              ReLU-6           [-1, 12, 24, 24]               0
            Conv2d-7           [-1, 16, 22, 22]           1,728
       BatchNorm2d-8           [-1, 16, 22, 22]              32
              ReLU-9           [-1, 16, 22, 22]               0
           Conv2d-10           [-1, 20, 20, 20]           2,880
      BatchNorm2d-11           [-1, 20, 20, 20]              40
             ReLU-12           [-1, 20, 20, 20]               0
         MaxPool2d-13           [-1, 20, 10, 10]               0
           Conv2d-14           [-1, 12, 10, 10]             252
      BatchNorm2d-15           [-1, 12, 10, 10]              24
             ReLU-16           [-1, 12, 10, 10]               0
           Conv2d-17           [-1, 16, 10, 10]           1,728
      BatchNorm2d-18           [-1, 16, 10, 10]              32
             ReLU-19           [-1, 16, 10, 10]               0
           Conv2d-20           [-1, 16, 10, 10]           2,304
      BatchNorm2d-21           [-1, 16, 10, 10]              32
             ReLU-22           [-1, 16, 10, 10]               0
           Conv2d-23           [-1, 16, 10, 10]           2,304
      BatchNorm2d-24           [-1, 16, 10, 10]              32
             ReLU-25           [-1, 16, 10, 10]               0
AdaptiveAvgPool2d-26             [-1, 16, 1, 1]               0
           Conv2d-27             [-1, 10, 1, 1]             170
================================================================
Total params: 12,534
Trainable params: 12,534
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.80
Params size (MB): 0.05
Estimated Total Size (MB): 0.85
----------------------------------------------------------------
```

### Parameter Distribution
```
Model Parameter Summary:
----------------------------------------
block1 Parameters: 88
block2 Parameters: 888
block3 Parameters: 1,760
block4 Parameters: 2,920
block5 Parameters: 0
block6 Parameters: 276
block7 Parameters: 1,760
block8 Parameters: 2,336
block9 Parameters: 2,336
block10 Parameters: 0
block11 Parameters: 170
----------------------------------------
Total Parameters: 12,534
----------------------------------------
```

## Training Progress
```
Epoch 1  - Loss: 1.9699, Accuracy: 38.86%, Test: 70.76%
Epoch 2  - Loss: 1.2410, Accuracy: 84.32%, Test: 96.23%
Epoch 3  - Loss: 0.7078, Accuracy: 96.79%, Test: 97.84%
Epoch 5  - Loss: 0.5847, Accuracy: 98.37%, Test: 98.74%
Epoch 8  - Loss: 0.5599, Accuracy: 98.89%, Test: 99.32%
Epoch 10 - Loss: 0.5513, Accuracy: 99.12%, Test: 99.41%
```

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
