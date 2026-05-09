"""
Threshold Tuning & PR Curve Analysis for Fall Detection

Tìm ngưỡng (threshold) tối ưu cho F1-Score cao nhất.
Vì mô hình dùng Focal Loss, ngưỡng tối ưu hiếm khi là 0.5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, f1_score, auc, roc_curve, roc_auc_score
from torch.utils.data import DataLoader

from src.data_pipeline.upfall_dataset import UPFallWindowDataset
from src.models.pose_model import TwoCamPoseClassifier
from src.utils.config import load_yaml


def evaluate_threshold(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_two_cameras: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect predictions (probabilities) and labels from validation set.
    Returns:
        probs: Predicted probabilities [0, 1]
        labels: True labels [0, 1]
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device).float()  # (B,)
            all_labels.append(labels.cpu().numpy())

            if use_two_cameras:
                pose_cam1 = batch.get("pose_cam1")
                pose_cam2 = batch.get("pose_cam2")
                if pose_cam1 is not None:
                    pose_cam1 = pose_cam1.to(device)
                if pose_cam2 is not None:
                    pose_cam2 = pose_cam2.to(device)
                logits, _ = model(pose_cam1, pose_cam2)
            else:
                pose_cam1 = batch.get("pose_cam1")
                if pose_cam1 is not None:
                    pose_cam1 = pose_cam1.to(device)
                logits, _ = model(pose_cam1, None)

            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()  # (B,)
            all_probs.append(probs)

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return probs, labels


def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> dict:
    """
    Tìm ngưỡng tối ưu bằng cách vẽ PR Curve và F1 Curve.
    """
    thresholds = np.arange(0.05, 1.0, 0.01)
    f1_scores = []
    precisions = []
    recalls = []

    for th in thresholds:
        preds = (probs >= th).astype(int)
        f1 = f1_score(labels, preds)
        p = (preds[labels == 1] == 1).sum() / max(preds.sum(), 1)  # True Positives / Predicted Positives
        r = (preds[labels == 1] == 1).sum() / max(labels.sum(), 1)  # True Positives / Actual Positives
        f1_scores.append(f1)
        precisions.append(p)
        recalls.append(r)

    f1_scores = np.array(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "thresholds": thresholds,
        "f1_scores": f1_scores,
        "precisions": np.array(precisions),
        "recalls": np.array(recalls),
    }


def plot_curves(results: dict, output_dir: Path) -> None:
    """Vẽ PR Curve, F1 Curve, ROC Curve."""
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = results["thresholds"]
    f1_scores = results["f1_scores"]
    precisions = results["precisions"]
    recalls = results["recalls"]
    best_threshold = results["best_threshold"]
    best_f1 = results["best_f1"]

    # Figure 1: F1 Score vs Threshold
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_scores, "b-", linewidth=2, label="F1 Score")
    ax.axvline(best_threshold, color="r", linestyle="--", linewidth=2, label=f"Optimal Threshold: {best_threshold:.3f}")
    ax.axhline(best_f1, color="g", linestyle="--", linewidth=1, alpha=0.7, label=f"Best F1: {best_f1:.4f}")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("F1 Score vs Decision Threshold", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_dir / "f1_vs_threshold.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recalls, precisions, "b-", linewidth=2, label="PR Curve")
    ax.axvline(np.interp(best_threshold, thresholds, recalls), color="r", linestyle="--", label=f"Threshold: {best_threshold:.3f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.savefig(output_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[✓] Saved curves to: {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    ap.add_argument("--split", default="2-10", help="Subject range for val/test (e.g., 2-10)")
    ap.add_argument("--use-two-cameras", action="store_true", default=True)
    ap.add_argument("--output-dir", default="eval_threshold_output")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    project_root = config_path.resolve().parent.parent
    processed_dir = project_root / Path(cfg["paths"]["processed_dir"])

    manifest_path = processed_dir / "synced_windows" / "upfall_windows.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    # Parse subject range
    start_s, end_s = map(int, args.split.split("-"))
    val_subjects = list(range(start_s, end_s + 1))

    # Load dataset (support legacy config keys)
    imu_len = None
    pose_len = None
    if "data" in cfg:
        imu_len = cfg["data"].get("imu_length")
        pose_len = cfg["data"].get("pose_length")
    if imu_len is None or pose_len is None:
        # fallback to upfall.target_lengths
        t = cfg.get("upfall", {}).get("target_lengths", {})
        imu_len = imu_len or t.get("imu")
        pose_len = pose_len or t.get("pose")

    if imu_len is None or pose_len is None:
        raise SystemExit("Could not determine imu/pose target lengths from config")

    val_ds = UPFallWindowDataset(
        manifest_path=str(manifest_path),
        subjects=val_subjects,
        imu_target_len=int(imu_len),
        pose_target_len=int(pose_len),
        require_pose=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    model = TwoCamPoseClassifier(
        # model/task config may live in train config; support fallbacks
        num_joints=int((cfg.get("model") or load_yaml(project_root / "configs/train_pose.yaml").get("model", {})).get("num_joints")),
        embed_dim=int((cfg.get("model") or load_yaml(project_root / "configs/train_pose.yaml").get("model", {})).get("embed_dim")),
        num_classes=int((cfg.get("task") or load_yaml(project_root / "configs/train_pose.yaml").get("task", {})).get("num_classes", 1)),
        backbone_type=str((cfg.get("model") or load_yaml(project_root / "configs/train_pose.yaml").get("model", {})).get("backbone", "temporal_cnn")),
        num_channels=int((cfg.get("model") or load_yaml(project_root / "configs/train_pose.yaml").get("model", {})).get("num_channels", 3)),
    ).to(device)

    # Load checkpoint and extract model state dict if wrapped
    raw_state = torch.load(ckpt_path, map_location=device)
    if isinstance(raw_state, dict):
        # support various checkpoint wrappers: 'model', 'model_state_dict', 'state_dict'
        if "model" in raw_state:
            state_dict = raw_state["model"]
        elif "model_state_dict" in raw_state:
            state_dict = raw_state["model_state_dict"]
        elif "state_dict" in raw_state:
            state_dict = raw_state["state_dict"]
        else:
            state_dict = raw_state
    else:
        state_dict = raw_state

    # Enforce strict loading so mismatches fail loudly instead of silently using random weights
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[✓] Successfully loaded checkpoint weights from: {ckpt_path}")
    except Exception as e:
        print("\n" + "!" * 60)
        print("ERROR: Model and checkpoint state_dict mismatch. Loading failed with strict=True")
        print(str(e))
        print("!" * 60 + "\n")
        raise

    # Evaluate
    probs, labels = evaluate_threshold(model, val_loader, device, use_two_cameras=args.use_two_cameras)

    # Find optimal threshold
    results = find_optimal_threshold(probs, labels)

    # Print results
    print("\n" + "=" * 60)
    print("THRESHOLD TUNING RESULTS")
    print("=" * 60)
    print(f"Best Threshold:  {results['best_threshold']:.4f}")
    print(f"Best F1 Score:   {results['best_f1']:.4f}")
    print(f"Subjects:        {val_subjects}")
    print(f"Total Samples:   {len(labels)}")
    print(f"Positive (Fall): {labels.sum()} ({100*labels.sum()/len(labels):.1f}%)")
    print(f"Negative (ADL):  {(1-labels).sum()} ({100*(1-labels).sum()/len(labels):.1f}%)")
    print("=" * 60)

    # Evaluate at default threshold 0.5
    preds_default = (probs >= 0.5).astype(int)
    f1_default = f1_score(labels, preds_default)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    print(f"\nMetrics at Threshold = 0.5:")
    print(f"  F1-Score: {f1_default:.4f}")
    print(f"  Accuracy: {accuracy_score(labels, preds_default):.4f}")
    print(f"  Precision: {precision_score(labels, preds_default):.4f}")
    print(f"  Recall: {recall_score(labels, preds_default):.4f}")

    # Evaluate at optimal threshold
    preds_optimal = (probs >= results["best_threshold"]).astype(int)
    print(f"\nMetrics at Optimal Threshold = {results['best_threshold']:.4f}:")
    print(f"  F1-Score: {results['best_f1']:.4f}")
    print(f"  Accuracy: {accuracy_score(labels, preds_optimal):.4f}")
    print(f"  Precision: {precision_score(labels, preds_optimal):.4f}")
    print(f"  Recall: {recall_score(labels, preds_optimal):.4f}")

    # Plot curves
    plot_curves(results, Path(args.output_dir))


if __name__ == "__main__":
    main()
