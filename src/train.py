import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from model import SimpleMNISTNet

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def calculate_dataset_statistics(dataset):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=4
    )

    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in tqdm(loader, desc="Calculating dataset statistics"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean.item(), std.item()

def train_and_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = SimpleMNISTNet().to(device)
    network.print_model_summary()

    # Calculate statistics using only training data
    initial_transform = transforms.Compose([transforms.ToTensor()])
    temp_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=initial_transform
    )
    mean, std = calculate_dataset_statistics(temp_dataset)
    print(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")

    # Define transforms for both training and testing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    transform_train = transforms.Compose([
        transforms.RandomRotation((-5, 5)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    # Load training and test datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Use Adam optimizer with lower learning rate and adjusted parameters
    optimizer = optim.Adam(
        network.parameters(),
        lr=0.0001,  # Lower initial learning rate
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-5  # Reduced weight decay
    )

    # More gradual learning rate schedule
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,  # Lower max learning rate
        steps_per_epoch=len(train_loader),
        epochs=20,
        pct_start=0.2,  # Warm up for 20% of training
        div_factor=10.0,
        final_div_factor=100.0,
        anneal_strategy='cos'
    )

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Moderate gradient clipping
    max_grad_norm = 1.0             # Standard clipping threshold

    best_test_accuracy = 0
    target_accuracy = 99.4

    for epoch in range(20):
        # Training phase
        network.train()
        total_loss = 0
        total_correct = 0

        print(f"\nEpoch {epoch+1}/20")
        for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}'):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = network(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_correct += get_num_correct(outputs, labels)

        train_accuracy = total_correct / len(train_dataset) * 100
        avg_loss = total_loss / len(train_loader)

        # Testing phase
        network.eval()
        test_correct = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images, labels = images.to(device), labels.to(device)
                outputs = network(images)
                test_correct += get_num_correct(outputs, labels)

        test_accuracy = test_correct / len(test_dataset) * 100
        best_test_accuracy = max(best_test_accuracy, test_accuracy)

        print(f"Epoch {epoch+1}")
        print(f"Training - Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Testing  - Accuracy: {test_accuracy:.2f}%")
        print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")

        # Save model if it achieves target accuracy
        if test_accuracy >= target_accuracy:
            print(f"\nReached target accuracy of {target_accuracy}%!")
            torch.save(network.state_dict(), 'best_model.pth')
            break

    print("\nTraining completed!")
    print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")

if __name__ == "__main__":
    train_and_test()
