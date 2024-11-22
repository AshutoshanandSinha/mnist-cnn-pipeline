import glob

import pytest
import torch
from torchvision import datasets, transforms

from model import MNISTNet


def get_latest_model():
    model_files = glob.glob("mnist_model_*.pth")
    return max(model_files) if model_files else None


def test_model_architecture():
    model = MNISTNet()
    num_params = model.count_parameters()
    assert (
        num_params < 25000
    ), f"Model has {num_params} parameters, should be less than 25000"
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape is incorrect"


def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTNet().to(device)
    model_path = get_latest_model()
    assert model_path is not None, "No trained model found"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Accuracy is only {accuracy:.2f}%, should be > 95%"


if __name__ == "__main__":
    pytest.main([__file__])
