from datetime import datetime
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from model import SimpleMNISTNet  # Updated to use a simplified model


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Enhanced transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=10,  # Reduced rotation
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5  # Reduced shear
        ),
        transforms.RandomErasing(p=0.1),  # Reduced erasing
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128,  # Increased from 64 to 128
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = SimpleMNISTNet().to(device)
    print(f"Total parameters: {model.count_parameters()}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # Reduced smoothing

    # Adjusted optimizer settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.002,  # Reduced from 0.003
        weight_decay=0.001,  # Reduced weight decay
        betas=(0.9, 0.999)
    )

    # Custom learning rate schedule
    total_steps = len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    def get_lr(step):
        if step < warmup_steps:
            return 0.002 * (step / warmup_steps)  # Linear warmup
        else:
            # Even gentler cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.002 * (0.4 + 0.6 * (1 + math.cos(math.pi * progress)) / 2)  # minimum LR will be 40% of max_lr

    # Initialize tracking variables before the loop
    running_loss = 0.0
    total = 0
    correct = 0
    best_acc = 0.0

    # Training loop modifications
    progress_bar = tqdm(train_loader, desc='Training')
    for i, (images, labels) in enumerate(progress_bar):
        # Update learning rate
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # Added gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics with momentum
        running_loss = 0.95 * running_loss + 0.05 * loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        # Update progress bar with current learning rate
        progress_bar.set_postfix({
            'Loss': running_loss,
            'Accuracy': accuracy,
            'LR': optimizer.param_groups[0]['lr']
        })

        # Track best accuracy
        if accuracy > best_acc:
            best_acc = accuracy

        # In the training loop, after optimizer.zero_grad():
        if i == total_steps // 2:  # Midway through training
            # Reduce weight decay
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = 0.005

    training_loss = running_loss / total_steps
    print(f"Training Loss: {training_loss:.4f}, Final Accuracy: {accuracy:.2f}%, Best Accuracy: {best_acc:.2f}%")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'mnist_model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    return model_path


if __name__ == '__main__':
    train()
