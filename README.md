# MNIST Model Training Pipeline

This project implements a CNN model for MNIST digit classification with a complete training and testing pipeline.

## Project Structure

```
mnist-cnn-pipeline/
├── src/
│   ├── model.py         # Contains SimpleMNISTNet architecture
│   ├── train.py         # Training script with optimizations
│   └── test_model.py    # Test suite for model validation
├── requirements.txt
└── README.md
```

## Model Architecture

The project uses `SimpleMNISTNet`, a CNN architecture with:
- Residual blocks
- Batch normalization
- Dropout regularization
- Less than 25,000 parameters
- Target accuracy > 95% on MNIST test set

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python src/train.py
   ```

3. Test the model:
   ```bash
   python src/test_model.py
   ```

## Requirements

The project requires the following packages:
```
torch>=1.9.0
torchvision>=0.10.0
pytest>=6.2.0
numpy>=1.19.0
flake8>=3.9.0
black>=22.3.0
isort>=5.9.0
tqdm>=4.65.0
```

## Training Features

- AdamW optimizer with learning rate scheduling
- Data augmentation (rotation, translation, random erasing)
- Label smoothing
- Gradient clipping
- Learning rate warmup and cosine decay
- Automatic model checkpointing

## Testing

The test suite validates:
- Model architecture (parameter count < 25,000)
- Output shape correctness
- Model accuracy (>95% on test set)

## Model Artifacts

Trained models are saved with timestamp in the format:
```
mnist_model_YYYYMMDD_HHMMSS.pth
```

## License

MIT License
