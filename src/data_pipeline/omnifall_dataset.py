from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_pipeline.sampling import resample_pose
from src.utils.jsonl import read_jsonl


@dataclass(frozen=True)
class OmniItem:
    id: str
    pose_path: str
    label: int


def _to_item(obj: dict[str, Any]) -> OmniItem:
    return OmniItem(
        id=str(obj["id"]),
        pose_path=str(obj["pose_path"]),
        label=int(obj["label"]),
    )


@lru_cache(maxsize=512)
def _load_pose_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as npz:
        t = npz["t"].astype(np.float32)
        kpts = npz["kpts"].astype(np.float32)
        conf = npz.get("conf")
        if conf is not None:
            conf = conf.astype(np.float32)
        else:
            conf = np.ones((kpts.shape[0], kpts.shape[1]), dtype=np.float32)
    return t, kpts, conf


class OmniFallPoseDataset(Dataset):
    def __init__(self, manifest_path: str | Path, indices: list[int], pose_target_len: int):
        raw_items = read_jsonl(manifest_path)
        items = [_to_item(it) for it in raw_items]

        self.items: list[OmniItem] = []
        for i in indices:
            if i < 0 or i >= len(items):
                continue
            item = items[i]
            if not Path(item.pose_path).exists():
                continue
            self.items.append(item)

        if not self.items:
            raise ValueError("No OmniFall items after filtering")
        self.pose_target_len = int(pose_target_len)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        t, kpts, conf = _load_pose_npz(item.pose_path)
        if t.size == 0 or t[-1] <= t[0]:
            # Empty or invalid time range -> return zero pose with confidence=0
            pose = np.zeros((self.pose_target_len, kpts.shape[1], 3), dtype=np.float32)
        else:
            # concatenate conf as third channel
            kpts_with_conf = np.concatenate([kpts, conf[:, :, np.newaxis]], axis=2)
            pose = resample_pose(t, kpts_with_conf, float(t[0]), float(t[-1]), self.pose_target_len, mode="nearest")
        pose = np.transpose(pose, (2, 0, 1))  # (C=3, T, J)
        return {
            "pose": torch.from_numpy(pose),
            "label": torch.tensor(item.label, dtype=torch.long),
        }
