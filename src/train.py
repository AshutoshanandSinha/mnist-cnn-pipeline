from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import MNISTNet


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    model = MNISTNet().to(device)
    print(f"Total parameters: {model.count_parameters()}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=1,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy="cos",
    )
    total_step = len(train_loader)
    model.train()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if (i + 1) % 100 == 0:
            accuracy = 100 * correct / total
            print(
                f"Step [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%"
            )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"mnist_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    return model_path


if __name__ == "__main__":
    train()
