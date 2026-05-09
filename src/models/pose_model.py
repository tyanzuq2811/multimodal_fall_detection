from __future__ import annotations

import torch
from torch import nn

from src.models.pose_backbone import PoseBackbone, PoseClassifierHead


class TwoCamPoseClassifier(nn.Module):
    def __init__(
        self,
        num_joints: int,
        embed_dim: int,
        num_classes: int = 1,
        backbone_type: str = "temporal_cnn",
        num_channels: int = 3,
    ):
        super().__init__()
        self.backbone = PoseBackbone(
            num_joints=num_joints,
            embed_dim=embed_dim,
            backbone_type=backbone_type,
                    num_channels=num_channels,
        )
        self.head = PoseClassifierHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, pose_cam1: torch.Tensor, pose_cam2: torch.Tensor | None = None):
        logits1, z1 = self.forward_single(pose_cam1)
        if pose_cam2 is None:
            return logits1, z1
        logits2, z2 = self.forward_single(pose_cam2)
        logits = torch.maximum(logits1, logits2)
        z = 0.5 * (z1 + z2)
        return logits, z

    def forward_single(self, pose: torch.Tensor):
        z = self.backbone(pose)
        logits = self.head(z)
        return logits, z
