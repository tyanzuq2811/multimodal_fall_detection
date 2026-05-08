from __future__ import annotations

from typing import Literal


DeviceSpec = Literal["auto", "cpu", "cuda"]


def resolve_device(device: DeviceSpec):
    if device == "cpu":
        import torch

        return torch.device("cpu")
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
