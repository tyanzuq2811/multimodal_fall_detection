from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data_pipeline.upfall_dataset import UPFallWindowDataset
from src.engines.common import BinaryFocalLoss, compute_focal_alpha, evaluate_classifier, save_checkpoint
from src.models.pose_model import TwoCamPoseClassifier
from src.utils.config import load_yaml
from src.utils.device import resolve_device
from src.utils.seed import set_global_seed


def _forward_pose_logits(model: TwoCamPoseClassifier, batch, device: torch.device):
    p1 = batch["pose_cam1"].to(device)
    p2 = batch["pose_cam2"].to(device)
    logits1, _ = model.forward_single(p1)
    logits2, _ = model.forward_single(p2)
    return logits1.view(-1), logits2.view(-1)


def _forward_pose_eval(model: TwoCamPoseClassifier, batch, device: torch.device) -> torch.Tensor:
    l1, l2 = _forward_pose_logits(model, batch, device)
    return torch.maximum(l1, l2)


def _load_backbone_weights(model: TwoCamPoseClassifier, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    
    # Normalize state dict: handle different checkpoint formats
    backbone_state = {}
    for k, v in state.items():
        # Remove leading "backbone." if present
        k_norm = k.replace("backbone.", "", 1)
        # If key starts with "encoder.", wrap with "backbone."
        if k_norm.startswith("encoder."):
            backbone_state[f"backbone.{k_norm}"] = v
        elif k_norm.startswith("backbone."):
            # Already has backbone prefix
            backbone_state[k_norm] = v
        else:
            # Key might be just encoder.net.* without any wrapper
            # Try to add backbone prefix
            backbone_state[f"backbone.{k_norm}"] = v
    
    if not backbone_state:
        raise SystemExit(f"No backbone weights found in: {ckpt_path}")
    
    model.backbone.load_state_dict(backbone_state, strict=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_pose.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    project_root = cfg_path.resolve().parent.parent

    data_cfg = load_yaml(project_root / cfg["data_config"])
    processed_dir = project_root / Path(data_cfg["paths"]["processed_dir"])
    manifest_path = processed_dir / "synced_windows" / "upfall_windows.jsonl"

    train_subjects = [int(x) for x in data_cfg["split"]["train_subjects"]]
    val_subjects = [int(x) for x in data_cfg["split"]["val_subjects"]]

    imu_len = int(data_cfg["upfall"]["target_lengths"]["imu"])
    pose_len = int(data_cfg["upfall"]["target_lengths"]["pose"])

    set_global_seed(int(cfg["train"]["seed"]))
    device = resolve_device(str(cfg["train"]["device"]))

    train_ds = UPFallWindowDataset(
        manifest_path=manifest_path,
        subjects=train_subjects,
        imu_target_len=imu_len,
        pose_target_len=pose_len,
        require_pose=True,
    )
    val_ds = UPFallWindowDataset(
        manifest_path=manifest_path,
        subjects=val_subjects,
        imu_target_len=imu_len,
        pose_target_len=pose_len,
        require_pose=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    model = TwoCamPoseClassifier(
        num_joints=int(cfg["model"]["num_joints"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        num_classes=int(cfg["task"]["num_classes"]),
        backbone_type=str(cfg["model"].get("backbone", "temporal_cnn")),
        num_channels=int(cfg["model"].get("num_channels", 3)),
    ).to(device)

    pre_cfg = cfg.get("input_checkpoints", {})
    pre_path = pre_cfg.get("pose_pretrained")
    if pre_path:
        ckpt_path = (project_root / str(pre_path)).resolve()
        if ckpt_path.exists():
            _load_backbone_weights(model, ckpt_path)
            print(f"Loaded pretrained backbone: {ckpt_path}")
        else:
            print(f"[WARN] Pretrained backbone not found: {ckpt_path}")

    alpha = compute_focal_alpha([int(it.label) for it in train_ds.items]).to(device)
    criterion = BinaryFocalLoss(alpha=float(alpha.item()), gamma=2.0)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    best_f1 = -1.0
    weights_dir = project_root / str(cfg["output"]["weights_dir"])
    out_path = weights_dir / str(cfg["output"]["ckpt_name"])
    printed_shape = False

    for epoch in range(int(cfg["train"]["max_epochs"])):
        model.train()
        for batch in train_loader:
            if not printed_shape:
                print(
                    f"[shape] pose_cam1={tuple(batch['pose_cam1'].shape)} "
                    f"pose_cam2={tuple(batch['pose_cam2'].shape)}"
                )
                printed_shape = True
            y = batch["label"].to(device).float()
            l1, l2 = _forward_pose_logits(model, batch, device)
            loss = 0.5 * (criterion(l1, y.view(-1)) + criterion(l2, y.view(-1)))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        val_res = evaluate_classifier(model, val_loader, device, criterion, _forward_pose_eval)
        print(f"[epoch {epoch+1}] val loss={val_res.loss:.4f} acc={val_res.acc:.4f} f1={val_res.f1:.4f}")
        if val_res.f1 > best_f1:
            best_f1 = val_res.f1
            save_checkpoint(model, out_path, extra={"config": cfg})
            print(f"  saved best -> {out_path}")


if __name__ == "__main__":
    main()
