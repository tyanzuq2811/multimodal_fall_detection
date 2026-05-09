from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.data_pipeline.upfall_dataset import UPFallWindowDataset
from src.models.pose_model import TwoCamPoseClassifier
from src.utils.config import load_yaml


def _resolve_model_cfg(project_root: Path, data_cfg: dict, train_cfg_path: Path) -> dict:
    model_cfg = dict(data_cfg.get("model") or {})
    task_cfg = dict(data_cfg.get("task") or {})
    if not model_cfg or not task_cfg:
        train_cfg = load_yaml(train_cfg_path)
        model_cfg = {**train_cfg.get("model", {}), **model_cfg}
        task_cfg = {**train_cfg.get("task", {}), **task_cfg}
    return {
        "num_joints": int(model_cfg.get("num_joints", 17)),
        "embed_dim": int(model_cfg.get("embed_dim", 128)),
        "num_classes": int(task_cfg.get("num_classes", 1)),
        "backbone": str(model_cfg.get("backbone", "temporal_cnn")),
        "num_channels": int(model_cfg.get("num_channels", 3)),
    }


def _load_state_dict(ckpt_path: Path, device: torch.device):
    raw_state = torch.load(ckpt_path, map_location=device)
    if isinstance(raw_state, dict):
        if "model" in raw_state:
            return raw_state["model"]
        if "model_state_dict" in raw_state:
            return raw_state["model_state_dict"]
        if "state_dict" in raw_state:
            return raw_state["state_dict"]
    return raw_state


def _parse_split(split: str) -> list[int]:
    split = split.strip()
    if "," in split:
        return [int(x) for x in split.split(",") if x.strip()]
    if "-" in split:
        start_s, end_s = split.split("-", 1)
        start_i = int(start_s)
        end_i = int(end_s)
        step = 1 if end_i >= start_i else -1
        return list(range(start_i, end_i + step, step))
    return [int(split)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time pose inference simulation for UP-Fall")
    ap.add_argument("--data-config", default="configs/data_config.yaml")
    ap.add_argument("--train-config", default="configs/train_pose.yaml")
    ap.add_argument("--ckpt", default="weights/pose_finetuned_upfall.pth")
    ap.add_argument("--split", default="11-12", help="Subjects to simulate, e.g. 11-12")
    ap.add_argument("--threshold", type=float, default=0.96, help="Decision threshold")
    ap.add_argument("--delay", type=float, default=0.0, help="Sleep between windows to simulate live stream")
    ap.add_argument("--limit", type=int, default=0, help="Max windows to simulate; 0 = all")
    ap.add_argument("--save-log", default="", help="Optional path to JSONL output")
    args = ap.parse_args()

    data_cfg_path = Path(args.data_config)
    train_cfg_path = Path(args.train_config)
    ckpt_path = Path(args.ckpt)

    data_cfg = load_yaml(data_cfg_path)
    project_root = data_cfg_path.resolve().parent.parent
    processed_dir = project_root / Path(data_cfg["paths"]["processed_dir"])
    manifest_path = processed_dir / "synced_windows" / "upfall_windows.jsonl"

    subjects = _parse_split(args.split)
    imu_len = int(data_cfg["upfall"]["target_lengths"]["imu"])
    pose_len = int(data_cfg["upfall"]["target_lengths"]["pose"])

    ds = UPFallWindowDataset(
        manifest_path=manifest_path,
        subjects=subjects,
        imu_target_len=imu_len,
        pose_target_len=pose_len,
        require_pose=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = _resolve_model_cfg(project_root, data_cfg, train_cfg_path)

    model = TwoCamPoseClassifier(
        num_joints=model_cfg["num_joints"],
        embed_dim=model_cfg["embed_dim"],
        num_classes=model_cfg["num_classes"],
        backbone_type=model_cfg["backbone"],
        num_channels=model_cfg["num_channels"],
    ).to(device)

    state_dict = _load_state_dict(ckpt_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    save_fp = None
    if args.save_log:
        save_path = Path(args.save_log)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_fp = save_path.open("w", encoding="utf-8")

    total = 0
    tp = fp = tn = fn = 0

    print(f"[✓] Loaded checkpoint: {ckpt_path}")
    print(f"[✓] Subjects: {subjects} | threshold={args.threshold:.3f} | windows={len(ds)}")
    print("=" * 80)

    with torch.no_grad():
        for idx, meta in enumerate(ds.items):
            if args.limit and idx >= args.limit:
                break

            sample = ds[idx]
            pose_cam1 = sample["pose_cam1"].unsqueeze(0).to(device)
            pose_cam2 = sample["pose_cam2"].unsqueeze(0).to(device)
            label = int(sample["label"].item())

            logits, _ = model(pose_cam1, pose_cam2)
            prob = float(torch.sigmoid(logits).view(-1).item())
            pred = int(prob >= args.threshold)

            outcome = "FALL" if pred else "ADL"
            truth = "FALL" if label else "ADL"
            status = "OK" if pred == label else "MISS"

            if label == 1 and pred == 1:
                tp += 1
            elif label == 0 and pred == 1:
                fp += 1
            elif label == 0 and pred == 0:
                tn += 1
            else:
                fn += 1
            total += 1

            line = (
                f"[S{meta.subject:02d} A{meta.activity:02d} T{meta.trial:02d} W{idx:06d}] "
                f"{meta.start_s:6.2f}s->{meta.end_s:6.2f}s | prob={prob:0.4f} | "
                f"pred={outcome:4s} | truth={truth:4s} | {status}"
            )
            print(line)

            if save_fp is not None:
                save_fp.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "subject": meta.subject,
                            "activity": meta.activity,
                            "trial": meta.trial,
                            "start_s": meta.start_s,
                            "end_s": meta.end_s,
                            "prob": prob,
                            "pred": pred,
                            "label": label,
                            "threshold": args.threshold,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                save_fp.flush()

            if args.delay > 0:
                time.sleep(args.delay)

    if save_fp is not None:
        save_fp.close()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / max(total, 1)

    print("=" * 80)
    print("REAL-TIME DEMO SUMMARY")
    print(f"Windows processed: {total}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Accuracy={acc:.4f} Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()