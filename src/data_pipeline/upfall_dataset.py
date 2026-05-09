from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_pipeline.sampling import resample_pose, resample_time_series
from src.utils.jsonl import read_jsonl


@lru_cache(maxsize=256)
def _load_imu_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as npz:
        t = npz["t"].astype(np.float32)
        x = npz["x"].astype(np.float32)
    return t, x


@lru_cache(maxsize=256)
def _load_pose_npz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as npz:
        t = npz["t"].astype(np.float32)
        kpts = npz["kpts"].astype(np.float32)  # (N, J, 2)
        conf = npz.get("conf")  # (N, J) - YOLO confidence
        if conf is not None:
            conf = conf.astype(np.float32)
        else:
            # If conf not available, use ones (full confidence)
            conf = np.ones((kpts.shape[0], kpts.shape[1]), dtype=np.float32)
    return t, kpts, conf


@dataclass(frozen=True)
class WindowSample:
    id: str
    label: int
    subject: int
    activity: int
    trial: int
    start_s: float
    end_s: float
    imu_path: str
    pose_cam1_path: str
    pose_cam2_path: str


def _to_window_sample(item: dict[str, Any]) -> WindowSample:
    return WindowSample(
        id=str(item["id"]),
        label=int(item["label"]),
        subject=int(item["subject"]),
        activity=int(item["activity"]),
        trial=int(item["trial"]),
        start_s=float(item["start_s"]),
        end_s=float(item["end_s"]),
        imu_path=str(item["imu_path"]),
        pose_cam1_path=str(item.get("pose_cam1_path", "")),
        pose_cam2_path=str(item.get("pose_cam2_path", "")),
    )


class UPFallWindowDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        subjects: list[int],
        imu_target_len: int,
        pose_target_len: int,
        require_pose: bool,
    ):
        self.manifest_path = str(manifest_path)
        self.subjects = set(int(s) for s in subjects)
        self.imu_target_len = int(imu_target_len)
        self.pose_target_len = int(pose_target_len)
        self.require_pose = bool(require_pose)

        raw_items = read_jsonl(self.manifest_path)
        self.items: list[WindowSample] = []
        for it in raw_items:
            ws = _to_window_sample(it)
            if ws.subject not in self.subjects:
                continue
            if self.require_pose:
                if not ws.pose_cam1_path or not ws.pose_cam2_path:
                    continue
                if not Path(ws.pose_cam1_path).exists() or not Path(ws.pose_cam2_path).exists():
                    continue
            self.items.append(ws)

        if not self.items:
            raise ValueError(f"No samples after filtering. manifest={self.manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        t_imu, x_imu = _load_imu_npz(it.imu_path)
        imu = resample_time_series(t_imu, x_imu, it.start_s, it.end_s, self.imu_target_len, mode="linear")
        imu_t = torch.from_numpy(imu.T)  # (6, T)

        out = {
            "id": it.id,
            "label": torch.tensor(it.label, dtype=torch.long),
            "imu": imu_t,
            "pose_cam1": None,
            "pose_cam2": None,
        }

        if it.pose_cam1_path and Path(it.pose_cam1_path).exists():
            t_p1, k1, c1 = _load_pose_npz(it.pose_cam1_path)
            # Concatenate kpts (N,J,2) and conf (N,J,1) -> (N,J,3)
            kpts_with_conf = np.concatenate([k1, c1[:, :, np.newaxis]], axis=2)
            p1 = resample_pose(t_p1, kpts_with_conf, it.start_s, it.end_s, self.pose_target_len, mode="nearest")
            p1 = np.transpose(p1, (2, 0, 1))  # (3, T, J) - channels: x, y, confidence
            out["pose_cam1"] = torch.from_numpy(p1)
        if it.pose_cam2_path and Path(it.pose_cam2_path).exists():
            t_p2, k2, c2 = _load_pose_npz(it.pose_cam2_path)
            kpts_with_conf = np.concatenate([k2, c2[:, :, np.newaxis]], axis=2)
            p2 = resample_pose(t_p2, kpts_with_conf, it.start_s, it.end_s, self.pose_target_len, mode="nearest")
            p2 = np.transpose(p2, (2, 0, 1))
            out["pose_cam2"] = torch.from_numpy(p2)

        return out
