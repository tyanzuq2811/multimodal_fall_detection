from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.metrics import accuracy, f1_binary


@dataclass(frozen=True)
class EvalResult:
    loss: float
    acc: float
    f1: float


def compute_class_weights(labels: list[int], num_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_pos_weight(labels: list[int]) -> torch.Tensor:
    arr = np.asarray(labels, dtype=np.int64)
    pos = float((arr == 1).sum())
    neg = float((arr == 0).sum())
    if pos <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(neg / pos, dtype=torch.float32)


@torch.no_grad()
def evaluate_classifier(
    model,
    loader: DataLoader,
    device: torch.device,
    criterion,
    forward_fn,
) -> EvalResult:
    def _set_eval(m):
        if isinstance(m, torch.nn.Module):
            m.eval()
            return
        if isinstance(m, (tuple, list)):
            for sub in m:
                _set_eval(sub)

    _set_eval(model)
    losses: list[float] = []
    ys: list[int] = []
    ps: list[int] = []

    for batch in loader:
        y = batch["label"].to(device)
        logits = forward_fn(model, batch, device)
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[-1] == 1):
            logits_1 = logits.view(-1)
            y_float = y.float().view(-1)
            loss = criterion(logits_1, y_float)
            losses.append(float(loss.item()))
            probs = torch.sigmoid(logits_1)
            pred = (probs >= 0.5).long()
        else:
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            pred = torch.argmax(logits, dim=-1)

        ys.extend(y.long().detach().cpu().numpy().tolist())
        ps.extend(pred.detach().cpu().numpy().tolist())

    y_arr = np.asarray(ys, dtype=np.int64)
    p_arr = np.asarray(ps, dtype=np.int64)
    return EvalResult(
        loss=float(np.mean(losses) if losses else 0.0),
        acc=accuracy(y_arr, p_arr),
        f1=f1_binary(y_arr, p_arr, pos_label=1),
    )


def save_checkpoint(model: torch.nn.Module, out_path: str | Path, extra: dict[str, Any] | None = None) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, p)
