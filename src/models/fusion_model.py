from __future__ import annotations

import torch
from torch import nn


class FusionMLP(nn.Module):
    def __init__(self, imu_embed_dim: int, pose_embed_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(imu_embed_dim + pose_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, imu_z: torch.Tensor, pose_z: torch.Tensor):
        z = torch.cat([imu_z, pose_z], dim=-1)
        return self.net(z)
