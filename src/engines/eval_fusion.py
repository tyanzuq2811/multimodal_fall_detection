from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.data_pipeline.upfall_dataset import UPFallWindowDataset
from src.models.fusion_mlp import FusionMLP
from src.models.imu_model import IMUClassifier
from src.models.pose_model import TwoCamPoseClassifier
from src.utils.config import load_yaml
from src.utils.device import resolve_device


def _apply_modality_weights(logits_triplet: torch.Tensor, weights: tuple[float, float, float]) -> torch.Tensor:
    w = torch.tensor(weights, dtype=logits_triplet.dtype, device=logits_triplet.device).view(1, 3)
    return logits_triplet * w


def _forward_fusion(
    models,
    batch,
    device: torch.device,
    modality_weights: tuple[float, float, float],
) -> torch.Tensor:
    imu_model, pose_model, fusion_model = models
    imu = batch["imu"].to(device)
    p1 = batch["pose_cam1"].to(device)
    p2 = batch["pose_cam2"].to(device)

    with torch.no_grad():
        imu_logits, _ = imu_model(imu)
        cam1_logits, _ = pose_model.forward_single(p1)
        cam2_logits, _ = pose_model.forward_single(p2)

    logits_triplet = torch.stack(
        [cam1_logits.view(-1), cam2_logits.view(-1), imu_logits.view(-1)],
        dim=-1,
    )
    logits_triplet = _apply_modality_weights(logits_triplet, modality_weights)
    logits = fusion_model(logits_triplet).view(-1)
    return logits


def _load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_fusion.yaml")
    ap.add_argument("--split", default="11-12", help="Subject range for eval (e.g., 11-12)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    ap.add_argument("--fusion-ckpt", default=None, help="Override fusion checkpoint path")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    project_root = cfg_path.resolve().parent.parent

    data_cfg = load_yaml(project_root / cfg["data_config"])
    processed_dir = project_root / Path(data_cfg["paths"]["processed_dir"])
    cfg_manifest = data_cfg.get("upfall", {}).get("manifest_path")
    if cfg_manifest:
        manifest_path = project_root / Path(cfg_manifest)
    else:
        manifest_path = processed_dir / "synced_windows" / "upfall_windows.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    start_s, end_s = map(int, args.split.split("-"))
    eval_subjects = list(range(start_s, end_s + 1))

    imu_len = int(data_cfg["upfall"]["target_lengths"]["imu"])
    pose_len = int(data_cfg["upfall"]["target_lengths"]["pose"])

    eval_ds = UPFallWindowDataset(
        manifest_path=manifest_path,
        subjects=eval_subjects,
        imu_target_len=imu_len,
        pose_target_len=pose_len,
        require_pose=True,
    )

    device = resolve_device(str(args.device))
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    imu_model = IMUClassifier(
        imu_channels=6,
        embed_dim=int(cfg["model"]["imu_embed_dim"]),
        num_classes=int(cfg["task"]["num_classes"]),
    )
    pose_model = TwoCamPoseClassifier(
        num_joints=int(data_cfg["upfall"]["pose"]["num_joints"]),
        embed_dim=int(cfg["model"]["pose_embed_dim"]),
        num_classes=int(cfg["task"]["num_classes"]),
        backbone_type=str(cfg["model"].get("pose_backbone", "temporal_cnn")),
    )
    fusion_model = FusionMLP(
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        dropout=float(cfg["model"].get("dropout", 0.2)),
        num_classes=int(cfg["task"]["num_classes"]),
    )

    imu_ckpt = project_root / str(cfg["input_checkpoints"]["imu_ckpt"])
    pose_ckpt = project_root / str(cfg["input_checkpoints"]["pose_ckpt"])
    fusion_ckpt = (
        (project_root / args.fusion_ckpt)
        if args.fusion_ckpt
        else (project_root / str(cfg["output"]["weights_dir"]) / str(cfg["output"]["ckpt_name"]))
    )

    if not imu_ckpt.exists() or not pose_ckpt.exists() or not fusion_ckpt.exists():
        raise SystemExit(
            "Missing checkpoints. Expect imu="
            f"{imu_ckpt.exists()} pose={pose_ckpt.exists()} fusion={fusion_ckpt.exists()}."
        )

    _load_checkpoint(imu_model, imu_ckpt)
    _load_checkpoint(pose_model, pose_ckpt)
    _load_checkpoint(fusion_model, fusion_ckpt)

    imu_model.to(device).eval()
    pose_model.to(device).eval()
    fusion_model.to(device).eval()

    mw_cfg = cfg.get("model", {}).get("modality_weights", {})
    modality_weights = (
        float(mw_cfg.get("cam1", 1.0)),
        float(mw_cfg.get("cam2", 1.0)),
        float(mw_cfg.get("imu", 1.0)),
    )

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in eval_loader:
            labels = batch["label"].to(device).float()
            logits = _forward_fusion((imu_model, pose_model, fusion_model), batch, device, modality_weights)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    preds = (probs >= float(args.threshold)).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    print("\n" + "=" * 60)
    print("FUSION TEST RESULTS")
    print("=" * 60)
    print(f"Subjects: {eval_subjects}")
    print(f"Threshold: {float(args.threshold):.4f}")
    print(f"Total Samples: {len(labels)}")
    print(f"Positive (Fall): {labels.sum()} ({100*labels.sum()/len(labels):.1f}%)")
    print(f"Negative (ADL): {(1-labels).sum()} ({100*(1-labels).sum()/len(labels):.1f}%)")
    print("-")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("-")
    print(f"TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
    print("=" * 60)


if __name__ == "__main__":
    main()
