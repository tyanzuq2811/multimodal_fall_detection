from __future__ import annotations

from typing import Literal

import numpy as np


def resample_time_series(
    t: np.ndarray,
    x: np.ndarray,
    start_s: float,
    end_s: float,
    target_len: int,
    mode: Literal["linear", "nearest"] = "linear",
) -> np.ndarray:
    if target_len <= 0:
        raise ValueError("target_len must be > 0")
    if end_s <= start_s:
        raise ValueError("end_s must be > start_s")

    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != t.shape[0]:
        raise ValueError("x and t length mismatch")

    if t.shape[0] == 0:
        return np.zeros((target_len, x.shape[1]), dtype=np.float32)

    grid = np.linspace(start_s, end_s, num=target_len, endpoint=False, dtype=np.float64)
    grid = np.clip(grid, float(t[0]), float(t[-1]))

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "nearest":
        idx = np.searchsorted(t, grid, side="left")
        idx = np.clip(idx, 0, t.shape[0] - 1)
        out = x[idx]
        return out.astype(np.float32)

    # linear interpolation per-channel
    out = np.zeros((target_len, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(grid, t, x[:, c]).astype(np.float32)
    return out


def resample_pose(
    t: np.ndarray,
    kpts: np.ndarray,
    start_s: float,
    end_s: float,
    target_len: int,
    mode: Literal["linear", "nearest"] = "nearest",
) -> np.ndarray:
    # kpts: (N, J, C) where C is any number of channels (2 for (x,y), 3 for (x,y,conf), etc)
    kpts = np.asarray(kpts)
    if kpts.ndim != 3:
        raise ValueError("kpts must have shape (N, J, C)")
    n, j, c = kpts.shape
    flat = kpts.reshape(n, j * c)
    out = resample_time_series(t, flat, start_s, end_s, target_len, mode=mode)
    return out.reshape(target_len, j, c).astype(np.float32)
