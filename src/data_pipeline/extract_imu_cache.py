from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data_pipeline.upfall_csv import extract_accel_6d, read_upfall_csv
from src.data_pipeline.upfall_scan import iter_upfall_trials
from src.utils.config import load_yaml


def imu_cache_path(processed_dir: Path, subject: int, activity: int, trial: int) -> Path:
    return (
        processed_dir
        / "imu_windows"
        / f"Subject{subject}"
        / f"Activity{activity}"
        / f"Trial{trial}"
        / "imu_wrist_belt_6d.npz"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    project_root = config_path.resolve().parent.parent
    raw_upfall_dir = project_root / Path(cfg["paths"]["raw_upfall_dir"])
    processed_dir = project_root / Path(cfg["paths"]["processed_dir"])

    trials = iter_upfall_trials(raw_upfall_dir)
    if not trials:
        raise SystemExit(f"No UP-Fall trials found under: {raw_upfall_dir}")

    num_done = 0
    num_skipped = 0
    for t in trials:
        out_path = imu_cache_path(processed_dir, t.subject, t.activity, t.trial)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            num_skipped += 1
            continue

        df = read_upfall_csv(t.csv_path)
        t_sec, x = extract_accel_6d(df)

        np.savez_compressed(
            out_path,
            t=np.asarray(t_sec, dtype=np.float32),
            x=np.asarray(x, dtype=np.float32),
            subject=np.int32(t.subject),
            activity=np.int32(t.activity),
            trial=np.int32(t.trial),
        )
        num_done += 1

    print(f"IMU cache: wrote={num_done}, skipped={num_skipped}, total_trials={len(trials)}")


if __name__ == "__main__":
    main()
