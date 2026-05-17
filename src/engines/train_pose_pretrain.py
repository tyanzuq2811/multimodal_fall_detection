from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_pipeline.omnifall_dataset import OmniFallPoseDataset
from src.engines.common import BinaryFocalLoss, compute_focal_alpha, evaluate_classifier, save_checkpoint
from src.models.pose_model import TwoCamPoseClassifier
from src.utils.config import load_yaml
from src.utils.device import resolve_device
from src.utils.jsonl import read_jsonl
from src.utils.seed import set_global_seed
from src.utils.train_logger import TrainLogger


def _forward_pose(model: TwoCamPoseClassifier, batch, device: torch.device) -> torch.Tensor:
    pose = batch["pose"].to(device)
    logits, _ = model.forward_single(pose)
    return logits


def _make_split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_ratio)
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()
    return train_idx, val_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_pose_pretrain.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    project_root = cfg_path.resolve().parent.parent

    data_cfg = load_yaml(project_root / cfg["data_config"])
    manifest_path = project_root / Path(data_cfg["omnifall"]["manifest_path"])

    raw_items = read_jsonl(manifest_path)
    if not raw_items:
        raise SystemExit("OmniFall manifest is empty. Run hf_omnifall_loader.py first.")

    val_ratio = float(cfg["data"]["val_ratio"])
    train_idx, val_idx = _make_split_indices(len(raw_items), val_ratio, int(cfg["train"]["seed"]))

    pose_len = int(cfg["data"]["pose_target_len"])

    set_global_seed(int(cfg["train"]["seed"]))
    device = resolve_device(str(cfg["train"]["device"]))

    train_ds = OmniFallPoseDataset(manifest_path, train_idx, pose_target_len=pose_len)
    val_ds = OmniFallPoseDataset(manifest_path, val_idx, pose_target_len=pose_len)

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
                y = batch["label"].to(device).float()
                logits = _forward_pose(model, batch, device).view(-1)
                loss = criterion(logits, y.view(-1))
                train_losses.append(float(loss.item()))
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

            train_loss = float(np.mean(train_losses) if train_losses else 0.0)
            val_res = evaluate_classifier(model, val_loader, device, criterion, _forward_pose)
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
