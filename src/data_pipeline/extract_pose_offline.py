from __future__ import annotations

import argparse
import csv
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data_pipeline.build_upfall_manifest import pose_cache_path
from src.data_pipeline.upfall_scan import iter_upfall_trials
from src.data_pipeline.upfall_zip import list_frames
from src.utils.config import load_yaml


COCO17 = 17
MP33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def _read_csv_start_dt(csv_path: Path) -> datetime:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        # skip 2 header rows
        next(reader)
        next(reader)
        first = next(reader)
    if not first:
        raise ValueError(f"No data rows in: {csv_path}")
    return datetime.fromisoformat(first[0])


def _extract_pose_ultralytics(zip_path: Path, csv_start_dt: datetime, model_name: str, device: str, frame_stride: int = 1):
    try:
        import cv2
    except ImportError as e:
        raise SystemExit("Missing dependency: opencv-python") from e

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Missing dependency: ultralytics") from e

    model = YOLO(model_name)
    frames = list_frames(zip_path)

    # Pre-compute which frames to process based on stride
    indices_to_process = [i for i in range(len(frames.members)) if i % frame_stride == 0]
    
    t_sec = np.zeros((len(indices_to_process),), dtype=np.float32)
    kpts = np.zeros((len(indices_to_process), COCO17, 2), dtype=np.float32)
    conf = np.zeros((len(indices_to_process), COCO17), dtype=np.float32)

    with zipfile.ZipFile(frames.zip_path, "r") as zf:
        for out_i, i in enumerate(indices_to_process):
            member = frames.members[i]
            ts_dt = frames.timestamps[i]
            t_sec[out_i] = float((ts_dt - csv_start_dt).total_seconds())

            raw = zf.read(member)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                continue

            results = model.predict(source=img, verbose=False, device=device)
            if not results:
                continue
            r0 = results[0]
            if getattr(r0, "keypoints", None) is None:
                continue
            if r0.keypoints.xy is None or len(r0.keypoints.xy) == 0:
                continue

            xy = r0.keypoints.xy
            cf = getattr(r0.keypoints, "conf", None)
            xy = xy.detach().cpu().numpy()  # (n, 17, 2)
            if cf is not None:
                cf = cf.detach().cpu().numpy()  # (n, 17)
                pick = int(np.argmax(cf.mean(axis=1)))
                conf[out_i] = cf[pick].astype(np.float32)
            else:
                pick = 0

            pts = xy[pick].astype(np.float32)
            pts[:, 0] /= float(w)
            pts[:, 1] /= float(h)
            kpts[out_i] = pts

    return t_sec, kpts, conf


def _extract_pose_mediapipe(zip_path: Path, csv_start_dt: datetime, use_gpu: bool = True, frame_stride: int = 1):
    try:
        import cv2
    except ImportError as e:
        raise SystemExit("Missing dependency: opencv-python") from e

    try:
        import mediapipe as mp
    except ImportError as e:
        raise SystemExit("Missing dependency: mediapipe") from e

    frames = list_frames(zip_path)
    
    # Pre-compute which frames to process based on stride
    indices_to_process = [i for i in range(len(frames.members)) if i % frame_stride == 0]
    
    t_sec = np.zeros((len(indices_to_process),), dtype=np.float32)
    kpts = np.zeros((len(indices_to_process), COCO17, 2), dtype=np.float32)
    conf = np.zeros((len(indices_to_process), COCO17), dtype=np.float32)

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with zipfile.ZipFile(frames.zip_path, "r") as zf:
        for out_i, i in enumerate(indices_to_process):
            member = frames.members[i]
            ts_dt = frames.timestamps[i]
            t_sec[out_i] = float((ts_dt - csv_start_dt).total_seconds())

            raw = zf.read(member)
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks is None:
                continue

            lm = res.pose_landmarks.landmark
            for j, idx in enumerate(MP33_TO_COCO17):
                kpts[out_i, j, 0] = float(lm[idx].x)
                kpts[out_i, j, 1] = float(lm[idx].y)
                conf[out_i, j] = float(lm[idx].visibility)

    pose.close()
    return t_sec, kpts, conf


def _extract_pose_dummy(zip_path: Path, csv_start_dt: datetime, frame_stride: int = 1):
    frames = list_frames(zip_path)
    
    # Pre-compute which frames to process based on stride
    indices_to_process = [i for i in range(len(frames.members)) if i % frame_stride == 0]
    
    t_sec = np.array([(frames.timestamps[i] - csv_start_dt).total_seconds() for i in indices_to_process], dtype=np.float32)
    kpts = np.zeros((len(indices_to_process), COCO17, 2), dtype=np.float32)
    conf = np.zeros((len(indices_to_process), COCO17), dtype=np.float32)
    return t_sec, kpts, conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cpu", help="device: cpu, 0 (GPU:0), 0,1 (GPU:0,1) etc")
    ap.add_argument("--model", default="yolov8n-pose.pt")
    ap.add_argument("--frame-stride", type=int, default=1, help="process every Nth frame (e.g., 5 = every 5th frame)")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    project_root = config_path.resolve().parent.parent
    raw_upfall_dir = project_root / Path(cfg["paths"]["raw_upfall_dir"])
    processed_dir = project_root / Path(cfg["paths"]["processed_dir"])

    cameras = [int(x) for x in cfg["upfall"]["cameras"]]
    allowed_subjects = set(int(x) for x in cfg["upfall"].get("extract_subjects", []))
    backend = str(cfg["upfall"]["pose"].get("backend", "ultralytics"))

    trials = iter_upfall_trials(raw_upfall_dir)
    if not trials:
        raise SystemExit(f"No UP-Fall trials found under: {raw_upfall_dir}")

    wrote = 0
    skipped = 0

    for t in trials:
        if allowed_subjects and t.subject not in allowed_subjects:
            continue
        csv_start_dt = _read_csv_start_dt(t.csv_path)
        for cam_id in cameras:
            if cam_id not in t.camera_zips:
                continue
            zip_path = t.camera_zips[cam_id]
            out_path = pose_cache_path(processed_dir, t.subject, t.activity, t.trial, cam_id)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            try:
                if backend == "dummy":
                    t_sec, kpts, conf = _extract_pose_dummy(zip_path, csv_start_dt, frame_stride=args.frame_stride)
                elif backend == "ultralytics":
                    t_sec, kpts, conf = _extract_pose_ultralytics(
                        zip_path=zip_path,
                        csv_start_dt=csv_start_dt,
                        model_name=args.model,
                        device=args.device,
                        frame_stride=args.frame_stride,
                    )
                elif backend == "mediapipe":
                    t_sec, kpts, conf = _extract_pose_mediapipe(zip_path, csv_start_dt, use_gpu=(args.device != "cpu"), frame_stride=args.frame_stride)
                else:
                    raise SystemExit(f"Unknown pose backend: {backend}")

                np.savez_compressed(
                    out_path,
                    t=np.asarray(t_sec, dtype=np.float32),
                    kpts=np.asarray(kpts, dtype=np.float32),
                    conf=np.asarray(conf, dtype=np.float32),
                    subject=np.int32(t.subject),
                    activity=np.int32(t.activity),
                    trial=np.int32(t.trial),
                    camera=np.int32(cam_id),
                )
                wrote += 1
            except Exception as e:
                print(f"[SKIP] S{t.subject}-A{t.activity}-T{t.trial}-C{cam_id}: {type(e).__name__}: {e}")
                skipped += 1
                continue

    print(f"Pose cache: wrote={wrote}, skipped={skipped}")


if __name__ == "__main__":
    main()
