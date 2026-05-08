from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == pos_label) & (y_pred == pos_label)).sum())
    fp = int(((y_true != pos_label) & (y_pred == pos_label)).sum())
    fn = int(((y_true == pos_label) & (y_pred != pos_label)).sum())
    denom = (2 * tp + fp + fn)
    return float(0.0 if denom == 0 else (2 * tp) / denom)
