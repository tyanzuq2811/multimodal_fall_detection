from __future__ import annotations

import torch
from torch import nn

from src.models.imu_backbone import IMUBackbone, IMUClassifierHead


class IMUClassifier(nn.Module):
    def __init__(self, imu_channels: int, embed_dim: int, num_classes: int = 2):
        super().__init__()
        self.backbone = IMUBackbone(imu_channels=imu_channels, embed_dim=embed_dim)
        self.head = IMUClassifierHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, imu: torch.Tensor):
        z = self.backbone(imu)
        logits = self.head(z)
        return logits, z
