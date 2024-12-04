import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SimpleMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial feature extraction - no padding
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=0, bias=False),  # 28->26
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        # Deep feature extraction - no padding
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, padding=0, bias=False),  # 26->24
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, padding=0, bias=False),  # 24->22
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(16, 20, kernel_size=3, padding=0, bias=False),  # 22->20
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )

        # Spatial reduction and channel compression
        self.maxpool = nn.MaxPool2d(2, 2)  # 20->10
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(20, 12, kernel_size=1),  # Single channel reduction
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )

        # Feature refinement with 3x3 convs
        self.block5 = nn.Sequential(
            nn.Conv2d(12, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.block7 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.final_conv = nn.Conv2d(16, 10, kernel_size=1)

    def forward(self, x):
        # Initial features without padding
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Spatial reduction and channel compression
        x = self.maxpool(x)
        x = self.channel_reduce(x)

        # Feature refinement
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        # Classification
        x = self.gap(x)
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)

    def count_parameters(self) -> Dict[str, int]:
        params_count = {}

        # Count parameters for each block
        for i, block in enumerate([self.block1, self.block2, self.block3, self.block4, self.maxpool, self.channel_reduce, self.block5, self.block6, self.block7, self.gap, self.final_conv], 1):
            params_count[f'block{i}'] = sum(p.numel() for p in block.parameters())

        # Total parameters
        params_count['total'] = sum(p.numel() for p in self.parameters())
        return params_count

    def print_model_summary(self):
        params = self.count_parameters()

        print("\nModel Parameter Summary:")
        print("-" * 40)
        # Only iterate through the blocks we actually have
        for block_name in params:
            if block_name != 'total':  # Skip the total count for now
                print(f"{block_name} Parameters: {params[block_name]:,}")
        print("-" * 40)
        print(f"Total Parameters: {params['total']:,}")
        print("-" * 40)
