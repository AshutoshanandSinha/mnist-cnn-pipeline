import torch.nn as nn
import torch.nn.functional as F


class SimpleMNISTNet(nn.Module):
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()

        # Initial convolution with slightly more filters
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # First residual block with channel expansion
        self.conv2a = nn.Conv2d(16, 18, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(18)
        self.conv2b = nn.Conv2d(18, 18, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(18)
        self.conv2_shortcut = nn.Conv2d(16, 18, kernel_size=1)

        # Second residual block with small expansion
        self.conv3a = nn.Conv2d(18, 22, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(22)
        self.conv3b = nn.Conv2d(22, 22, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(22)
        self.conv3_shortcut = nn.Conv2d(18, 22, kernel_size=1)

        # Final convolution
        self.conv4 = nn.Conv2d(22, 30, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(30)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(30 * 2 * 2, 10)

        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        # Initial features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        # First residual block with shortcut
        identity = self.conv2_shortcut(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + identity)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        # Second residual block
        identity = self.conv3_shortcut(x)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = self.bn3b(self.conv3b(x))
        x = F.relu(x + identity)
        x = self.dropout(x)

        # Final convolution
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.avgpool(x)

        # Simplified classifier
        x = x.view(-1, 30 * 2 * 2)
        x = self.fc(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
