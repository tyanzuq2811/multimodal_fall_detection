from __future__ import annotations

import torch
from torch import nn

from src.models.temporal_cnn import TemporalConvEncoder


def _to_bctv(pose: torch.Tensor) -> torch.Tensor:
    # Accept (B, T, J, 2) or (B, 2, T, J)
    if pose.ndim != 4:
        raise ValueError("Expected pose shape (B, T, J, 2) or (B, 2, T, J)")
    if pose.shape[1] == 2:
        return pose
    if pose.shape[-1] == 2:
        return pose.permute(0, 3, 1, 2)
    raise ValueError("Pose tensor has invalid shape")


class TemporalPoseBackbone(nn.Module):
    def __init__(self, num_joints: int, embed_dim: int):
        super().__init__()
        in_channels = int(num_joints) * 2
        self.encoder = TemporalConvEncoder(in_channels=in_channels, embed_dim=embed_dim)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        x = _to_bctv(pose).permute(0, 2, 3, 1)  # (B, T, J, 2)
        b, t, j, c = x.shape
        x = x.reshape(b, t, j * c)
        return self.encoder(x)


class PoseBackbone(nn.Module):
    def __init__(
        self,
        num_joints: int,
        embed_dim: int,
        backbone_type: str = "temporal_cnn",
    ):
        super().__init__()
        if backbone_type != "temporal_cnn":
            raise ValueError(f"Unknown pose backbone: {backbone_type}")
        self.backbone = TemporalPoseBackbone(num_joints=num_joints, embed_dim=embed_dim)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.backbone(pose)


class PoseClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)
