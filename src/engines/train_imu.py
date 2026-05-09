from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data_pipeline.collate import collate_upfall_batch
from src.data_pipeline.upfall_dataset import UPFallWindowDataset
from src.engines.common import BinaryFocalLoss, compute_focal_alpha, evaluate_classifier, save_checkpoint
from src.models.imu_model import IMUClassifier
from src.utils.config import load_yaml
from src.utils.device import resolve_device
from src.utils.seed import set_global_seed


def _forward_imu(model: IMUClassifier, batch, device: torch.device) -> torch.Tensor:
    imu = batch["imu"].to(device).float()  # Shape: (B, 6, T) - [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    
    # ===== INSTANCE NORMALIZATION (Chuẩn hóa từng mẫu riêng biệt) =====
    # SỬA LỖI: Chỉ tính Mean/Std dọc theo trục Thời gian (dim=2) thôi, KHÔNG normalize theo batch
    # Điều này có ý nghĩa vật lý: triệt tiêu trọng trường (Gravity) và độ nghiêng cảm biến
    # Chỉ giữ lại Dynamic Acceleration - thứ duy nhất cần để bắt cú ngã
    
    accel = imu[:, :3, :]  # (B, 3, T)
    gyro = imu[:, 3:, :]   # (B, 3, T)
    
    # Instance normalization: mỗi sample được chuẩn hóa độc lập
    accel_mean = accel.mean(dim=2, keepdim=True)  # Shape: (B, 3, 1)
    accel_std = accel.std(dim=2, keepdim=True)    # Shape: (B, 3, 1)
    accel = (accel - accel_mean) / (accel_std + 1e-6)
    
    gyro_mean = gyro.mean(dim=2, keepdim=True)    # Shape: (B, 3, 1)
    gyro_std = gyro.std(dim=2, keepdim=True)      # Shape: (B, 3, 1)
    gyro = (gyro - gyro_mean) / (gyro_std + 1e-6)
    
    imu_norm = torch.cat([accel, gyro], dim=1)  # Reconstruct: (B, 6, T)
    
    logits, _ = model(imu_norm)
    return logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_imu.yaml")
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
        require_pose=False,
    )
    val_ds = UPFallWindowDataset(
        manifest_path=manifest_path,
        subjects=val_subjects,
        imu_target_len=imu_len,
        pose_target_len=pose_len,
        require_pose=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_upfall_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_upfall_batch,
    )

    model = IMUClassifier(
        imu_channels=int(cfg["model"]["imu_channels"]),
        embed_dim=int(cfg["model"]["embed_dim"]),
        num_classes=int(cfg["task"]["num_classes"]),
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
    printed_shape = False

    for epoch in range(int(cfg["train"]["max_epochs"])):
        model.train()
        for batch in train_loader:
            if not printed_shape:
                print(f"[shape] imu={tuple(batch['imu'].shape)}")
                printed_shape = True
            y = batch["label"].to(device).float()
            logits = _forward_imu(model, batch, device).view(-1)
            loss = criterion(logits, y.view(-1))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        val_res = evaluate_classifier(model, val_loader, device, criterion, _forward_imu)
        print(f"[epoch {epoch+1}] val loss={val_res.loss:.4f} acc={val_res.acc:.4f} f1={val_res.f1:.4f}")
        if val_res.f1 > best_f1:
            best_f1 = val_res.f1
            save_checkpoint(model, out_path, extra={"config": cfg})
            print(f"  saved best -> {out_path}")


if __name__ == "__main__":
    main()
