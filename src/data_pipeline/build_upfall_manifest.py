from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data_pipeline.extract_imu_cache import imu_cache_path
from src.data_pipeline.upfall_scan import iter_upfall_trials
from src.utils.config import load_yaml
from src.utils.jsonl import write_jsonl


def pose_cache_path(
    processed_dir: Path,
    subject: int,
    activity: int,
    trial: int,
    camera_id: int,
    pose_root: Path | None = None,
) -> Path:
    root = pose_root if pose_root is not None else processed_dir / "pose_features" / "upfall"
    return (
        root
        / f"Subject{subject}"
        / f"Activity{activity}"
        / f"Trial{trial}"
        / f"camera{camera_id}_pose.npz"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    project_root = config_path.resolve().parent.parent
    raw_upfall_dir = project_root / Path(cfg["paths"]["raw_upfall_dir"])
    processed_dir = project_root / Path(cfg["paths"]["processed_dir"])
    cameras = [int(x) for x in cfg["upfall"]["cameras"]]
    allowed_subjects = set(int(x) for x in cfg["upfall"].get("extract_subjects", []))

    fall_acts = set(int(x) for x in cfg["upfall"]["labels"]["fall_activities"])
    non_fall_acts = set(int(x) for x in cfg["upfall"]["labels"]["non_fall_activities"])
    win_len = float(cfg["upfall"]["windows"]["length_s"])
    win_stride = float(cfg["upfall"]["windows"]["stride_s"])

    cfg_manifest = cfg.get("upfall", {}).get("manifest_path")
    if args.out:
        out_path = Path(args.out)
    elif cfg_manifest:
        out_path = project_root / Path(cfg_manifest)
    else:
        out_path = processed_dir / "synced_windows" / "upfall_windows.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trials = iter_upfall_trials(raw_upfall_dir)
    if not trials:
        raise SystemExit(f"No UP-Fall trials found under: {raw_upfall_dir}")

    items = []
    n_pos = 0
    n_neg = 0

    for t in trials:
        if allowed_subjects and t.subject not in allowed_subjects:
            continue
        if t.activity in fall_acts:
            label = 1
        elif t.activity in non_fall_acts:
            label = 0
        else:
            continue

        imu_path = imu_cache_path(processed_dir, t.subject, t.activity, t.trial)
        if not imu_path.exists():
            raise SystemExit(
                f"Missing IMU cache {imu_path}. Run: python -m src.data_pipeline.extract_imu_cache"
            )
        imu = np.load(imu_path)
        t_sec = imu["t"].astype(float)
        if t_sec.size < 2:
            continue
        duration = float(t_sec[-1] - t_sec[0])
        if duration < win_len:
            continue

        pose_root = cfg.get("upfall", {}).get("pose_cache_dir")
        pose_root = project_root / Path(pose_root) if pose_root else None
        pose_paths = {
            cid: pose_cache_path(processed_dir, t.subject, t.activity, t.trial, cid, pose_root=pose_root)
            for cid in cameras
        }
        # We don't require pose files at manifest-build time, to allow IMU-only training.

        num_windows = int((duration - win_len) // win_stride) + 1
        for wi in range(num_windows):
            start_s = wi * win_stride
            end_s = start_s + win_len
            window_id = f"S{t.subject}_A{t.activity}_T{t.trial}_W{wi:06d}"
            items.append(
                {
                    "id": window_id,
                    "subject": t.subject,
                    "activity": t.activity,
                    "trial": t.trial,
                    "label": label,
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "imu_path": str(imu_path.as_posix()),
                    "pose_cam1_path": str(pose_paths.get(1, "")),
                    "pose_cam2_path": str(pose_paths.get(2, "")),
                }
            )

        if label == 1:
            n_pos += num_windows
        else:
            n_neg += num_windows

    write_jsonl(out_path, items)
    print(f"Wrote manifest: {out_path}")
    print(f"Windows: total={len(items)}, pos={n_pos}, neg={n_neg}")
    if n_pos == 0:
        print("[WARN] No positive (fall) windows. Bạn cần tải Activity 7-11 để có dữ liệu té ngã.")


if __name__ == "__main__":
    main()
