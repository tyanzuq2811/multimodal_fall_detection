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
from src.utils.train_logger import TrainLogger


def _forward_pose_logits(model: TwoCamPoseClassifier, batch, device: torch.device):
    p1 = batch["pose_cam1"].to(device)
    p2 = batch["pose_cam2"].to(device)
    logits1, _ = model.forward_single(p1)
    logits2, _ = model.forward_single(p2)
    return logits1.view(-1), logits2.view(-1)


def _forward_pose_eval(model: TwoCamPoseClassifier, batch, device: torch.device) -> torch.Tensor:
    l1, l2 = _forward_pose_logits(model, batch, device)
    return torch.maximum(l1, l2)


def _load_pretrained_weights(model: TwoCamPoseClassifier, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_pose.yaml")
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
            _load_pretrained_weights(model, ckpt_path)
            print(f"Loaded pretrained weights: {ckpt_path}")
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

    log_cfg = cfg.get("logging", {})
    wandb_cfg = cfg.get("wandb", {})
    log_dir = project_root / str(log_cfg.get("log_dir", "logs"))
    run_name = str(log_cfg.get("run_name") or cfg["task"]["name"])
    logger = TrainLogger(
        log_dir=log_dir,
        run_name=run_name,
        save_csv=log_cfg.get("save_csv", True),
        save_json=log_cfg.get("save_json", True),
        wandb_cfg=wandb_cfg,
        run_config=cfg,
    )

    try:
        for epoch in range(int(cfg["train"]["max_epochs"])):
            model.train()
            train_losses: list[float] = []
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
                train_losses.append(float(loss.item()))
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

            train_loss = float(sum(train_losses) / len(train_losses)) if train_losses else 0.0
            val_res = evaluate_classifier(model, val_loader, device, criterion, _forward_pose_eval)
            print(
                f"[epoch {epoch+1}] train loss={train_loss:.4f} "
                f"val loss={val_res.loss:.4f} acc={val_res.acc:.4f} f1={val_res.f1:.4f}"
            )
            is_best = val_res.f1 > best_f1
            if is_best:
                best_f1 = val_res.f1
                save_checkpoint(model, out_path, extra={"config": cfg})
                print(f"  saved best -> {out_path}")

            logger.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_res.loss,
                    "val_acc": val_res.acc,
                    "val_f1": val_res.f1,
                    "lr": float(optim.param_groups[0]["lr"]),
                    "is_best": is_best,
                },
                step=epoch + 1,
            )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
