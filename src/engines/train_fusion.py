from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data_pipeline.upfall_dataset import UPFallWindowDataset
from src.engines.common import compute_pos_weight, evaluate_classifier, save_checkpoint
from src.models.fusion_mlp import FusionMLP
from src.models.imu_model import IMUClassifier
from src.models.pose_model import TwoCamPoseClassifier
from src.utils.config import load_yaml
from src.utils.device import resolve_device
from src.utils.seed import set_global_seed


def _apply_modality_dropout(logits_triplet: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0:
        return logits_triplet
    b = logits_triplet.shape[0]
    r = torch.rand((b,), device=logits_triplet.device)
    drop_mask = r < p
    if not drop_mask.any():
        return logits_triplet
    logits_triplet = logits_triplet.clone()
    drop_idx = torch.randint(0, 3, (b,), device=logits_triplet.device)
    for i in range(b):
        if drop_mask[i]:
            logits_triplet[i, drop_idx[i]] = 0.0
    return logits_triplet


def _apply_modality_weights(logits_triplet: torch.Tensor, weights: tuple[float, float, float]) -> torch.Tensor:
    w = torch.tensor(weights, dtype=logits_triplet.dtype, device=logits_triplet.device).view(1, 3)
    return logits_triplet * w


def _forward_fusion(
    models,
    batch,
    device: torch.device,
    drop_p: float,
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
    if fusion_model.training:
        logits_triplet = _apply_modality_dropout(logits_triplet, drop_p)
    logits = fusion_model(logits_triplet).view(-1)
    return logits


def _load_checkpoint(model: nn.Module, path: Path) -> None:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_fusion.yaml")
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
    if not imu_ckpt.exists() or not pose_ckpt.exists():
        raise SystemExit(
            f"Missing checkpoints. Expect imu={imu_ckpt.exists()} pose={pose_ckpt.exists()}. "
            "Train IMU and Pose first."
        )
    _load_checkpoint(imu_model, imu_ckpt)
    _load_checkpoint(pose_model, pose_ckpt)

    imu_model.to(device).eval()
    pose_model.to(device).eval()
    for p in imu_model.parameters():
        p.requires_grad = False
    for p in pose_model.parameters():
        p.requires_grad = False

    fusion_model.to(device)
    train_labels = [int(it.label) for it in train_ds.items]
    pos_weight = compute_pos_weight(train_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(
        fusion_model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    mw_cfg = cfg.get("model", {}).get("modality_weights", {})
    modality_weights = (
        float(mw_cfg.get("cam1", 1.0)),
        float(mw_cfg.get("cam2", 1.0)),
        float(mw_cfg.get("imu", 1.0)),
    )
    print(
        "[fusion] modality_weights "
        f"cam1={modality_weights[0]:.3f} "
        f"cam2={modality_weights[1]:.3f} "
        f"imu={modality_weights[2]:.3f}"
    )

    best_f1 = -1.0
    weights_dir = project_root / str(cfg["output"]["weights_dir"])
    out_path = weights_dir / str(cfg["output"]["ckpt_name"])
    printed_shape = False

    models = (imu_model, pose_model, fusion_model)
    for epoch in range(int(cfg["train"]["max_epochs"])):
        fusion_model.train()
        for batch in train_loader:
            if not printed_shape:
                print(
                    f"[shape] imu={tuple(batch['imu'].shape)} "
                    f"pose_cam1={tuple(batch['pose_cam1'].shape)} "
                    f"pose_cam2={tuple(batch['pose_cam2'].shape)}"
                )
                printed_shape = True
            y = batch["label"].to(device).float()
            logits = _forward_fusion(
                models,
                batch,
                device,
                float(cfg["train"].get("modality_dropout_p", 0.0)),
                modality_weights,
            )
            loss = criterion(logits, y.view(-1))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        val_res = evaluate_classifier(
            models,
            val_loader,
            device,
            criterion,
            lambda m, b, d: _forward_fusion(m, b, d, 0.0, modality_weights),
        )
        print(f"[epoch {epoch+1}] val loss={val_res.loss:.4f} acc={val_res.acc:.4f} f1={val_res.f1:.4f}")
        if val_res.f1 > best_f1:
            best_f1 = val_res.f1
            save_checkpoint(fusion_model, out_path, extra={"config": cfg})
            print(f"  saved best -> {out_path}")


if __name__ == "__main__":
    main()
