from __future__ import annotations

import torch
from torch import nn


class TemporalConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        hidden_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        if x.ndim != 3:
            raise ValueError("Expected x shape (B, T, C)")
        x = x.transpose(1, 2)  # (B, C, T)
        z = self.net(x).squeeze(-1)  # (B, D)
        return z
