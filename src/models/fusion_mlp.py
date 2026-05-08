from __future__ import annotations

import torch
from torch import nn


class FusionMLP(nn.Module):
    def __init__(self, hidden_dim: int = 16, dropout: float = 0.2, num_classes: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, logits_triplet: torch.Tensor) -> torch.Tensor:
        # logits_triplet: (B, 3) = [logit_cam1, logit_cam2, logit_imu]
        if logits_triplet.ndim != 2 or logits_triplet.shape[1] != 3:
            raise ValueError("Expected logits_triplet shape (B, 3)")
        return self.net(logits_triplet)
