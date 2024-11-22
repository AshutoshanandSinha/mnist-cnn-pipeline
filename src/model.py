import torch.nn as nn


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()

        # First conv block - keep channels small initially
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # Second conv block - moderate increase in channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # Efficient feature mixer
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.relu3 = nn.ReLU()

        # Global average pooling instead of large FC layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Small FC layer
        self.fc = nn.Linear(24, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Feature mixer
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Global pooling and final classification
        x = self.global_pool(x)
        x = x.view(-1, 24)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
