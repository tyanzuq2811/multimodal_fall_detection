from __future__ import annotations

import torch
from torch import nn


class IMUBackbone(nn.Module):
    def __init__(self, imu_channels: int, embed_dim: int):
        super().__init__()
        self.imu_channels = int(imu_channels)
        self.net = nn.Sequential(
            nn.Conv1d(self.imu_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        # imu: (B, 6, T) or (B, T, 6)
        if imu.ndim != 3:
            raise ValueError("Expected imu shape (B, 6, T) or (B, T, 6)")
        if imu.shape[1] == self.imu_channels:
            x = imu
        elif imu.shape[2] == self.imu_channels:
            x = imu.transpose(1, 2)
        else:
            raise ValueError("imu channels mismatch")
        z = self.net(x).squeeze(-1)
        return z


class IMUClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)
