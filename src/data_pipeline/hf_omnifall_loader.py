from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.utils.config import load_yaml
from src.utils.jsonl import write_jsonl


MP33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def _resolve_paths(config_path: Path):
    cfg = load_yaml(config_path)
    project_root = config_path.resolve().parent.parent
    raw_dir = project_root / Path(cfg["paths"]["raw_omnifall_dir"])
    pose_dir = project_root / Path(cfg["omnifall"]["pose_dir"])
    manifest_path = project_root / Path(cfg["omnifall"]["manifest_path"])
    return cfg, project_root, raw_dir, pose_dir, manifest_path


def _normalize_label(value: Any, fall_values: set[str], non_fall_values: set[str], label_names: list[str] | None):
    if value is None:
        return None
    if label_names is not None and isinstance(value, (int, float)):
        idx = int(value)
        if 0 <= idx < len(label_names):
            value = label_names[idx]
    if isinstance(value, (int, float)) and int(value) in (0, 1):
        return int(value)
    s = str(value)
    if s in fall_values:
        return 1
    if s in non_fall_values:
        return 0
    if "*" in non_fall_values:
        return 0
    return None


def _build_repo_index(repo_id: str):
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        return None

    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    files = list_repo_files(repo_id, repo_type="dataset")
    media_files = [f for f in files if Path(f).suffix.lower() in exts]
    full_map = {f: f for f in media_files}
    stem_map: dict[str, str] = {}
    for f in media_files:
        stem = str(Path(f).with_suffix(""))
        stem_map.setdefault(stem, f)
    return {"full": full_map, "stem": stem_map, "exts": exts}


def _find_repo_archive(repo_id: str) -> str | None:
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        return None
    files = list_repo_files(repo_id, repo_type="dataset")
    for ext in (".tar", ".tar.gz", ".zip"):
        for f in files:
            if f.lower().endswith(ext):
                return f
    return None


def _ensure_media_cache(repo_id: str, raw_dir: Path) -> Path | None:
    media_dir = raw_dir / "media"
    marker = media_dir / ".extracted"
    if marker.exists():
        return media_dir

    archive = _find_repo_archive(repo_id)
    if archive is None:
        return None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    media_dir.mkdir(parents=True, exist_ok=True)
    archive_path = hf_hub_download(
        repo_id=repo_id,
        filename=archive,
        repo_type="dataset",
        cache_dir=str(raw_dir),
    )

    if archive_path.lower().endswith((".tar", ".tar.gz")):
        import tarfile

        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(media_dir)
    elif archive_path.lower().endswith(".zip"):
        import zipfile

        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(media_dir)
    else:
        return None

    marker.write_text(archive, encoding="utf-8")
    return media_dir


def _build_local_media_index(media_dir: Path):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    full_map: dict[str, Path] = {}
    stem_map: dict[str, Path] = {}
    base_map: dict[str, Path] = {}
    label_map: dict[str, list[Path]] = {}
    for p in media_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(media_dir).as_posix()
        full_map[rel] = p
        stem = str(Path(rel).with_suffix(""))
        stem_map.setdefault(stem, p)
        base = Path(rel).stem
        base_map.setdefault(base, p)
        parts = rel.split("/", 1)
        if parts:
            label = parts[0]
            label_map.setdefault(label, []).append(p)
    return {"full": full_map, "stem": stem_map, "base": base_map, "exts": exts, "labels": label_map}


def _resolve_local_path(rel_path: str, local_index: dict[str, Any] | None) -> Path | None:
    if local_index is None:
        return None
    rel = rel_path.lstrip("/")
    full_map = local_index["full"]
    stem_map = local_index["stem"]
    base_map = local_index["base"]
    exts = local_index["exts"]

    if rel in full_map:
        return full_map[rel]
    if rel in stem_map:
        return stem_map[rel]

    for ext in exts:
        cand = f"{rel}{ext}"
        if cand in full_map:
            return full_map[cand]

    prefixes = ["videos/", "video/", "data/", "raw/"]
    for prefix in prefixes:
        base = f"{prefix}{rel}"
        if base in full_map:
            return full_map[base]
        if base in stem_map:
            return stem_map[base]
        for ext in exts:
            cand = f"{base}{ext}"
            if cand in full_map:
                return full_map[cand]

    base = Path(rel).name
    if base in base_map:
        return base_map[base]
    return None


def _label_to_folder(label_name: str) -> str:
    mapping = {
        "walk": "walking",
        "fall": "fall",
        "fallen": "fallen",
        "sit_down": "sit_down",
        "sitting": "sitting",
        "lie_down": "lie_down",
        "lying": "lying",
        "stand_up": "stand_up",
        "standing": "standing",
        "other": "other",
        "kneel_down": "other",
        "kneeling": "other",
        "squat_down": "other",
        "squatting": "other",
        "crawl": "other",
        "jump": "other",
    }
    return mapping.get(label_name, "other")


def _resolve_repo_path(rel_path: str, repo_index: dict[str, Any] | None) -> str | None:
    if repo_index is None:
        return None
    full_map = repo_index["full"]
    stem_map = repo_index["stem"]
    exts = repo_index["exts"]

    if rel_path in full_map:
        return full_map[rel_path]
    if rel_path in stem_map:
        return stem_map[rel_path]

    for ext in exts:
        cand = f"{rel_path}{ext}"
        if cand in full_map:
            return full_map[cand]

    prefixes = ["videos/", "video/", "data/", "raw/"]
    for prefix in prefixes:
        base = f"{prefix}{rel_path}"
        if base in full_map:
            return full_map[base]
        if base in stem_map:
            return stem_map[base]
        for ext in exts:
            cand = f"{base}{ext}"
            if cand in full_map:
                return full_map[cand]
    return None


def _get_video_path(
    sample: dict[str, Any],
    raw_dir: Path,
    dataset_name: str,
    repo_index: dict[str, Any] | None,
    path_field: str,
    local_index: dict[str, Any] | None,
    local_video_root: Path | None,
    label_name: str | None,
    sample_index: int,
) -> tuple[Path | None, bool]:
    video = sample.get("video")
    if isinstance(video, dict):
        if video.get("path"):
            return Path(video["path"]), False
        if video.get("bytes"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=raw_dir)
            tmp.write(video["bytes"])
            tmp.flush()
            tmp.close()
            return Path(tmp.name), False
    if isinstance(video, str):
        return Path(video), False

    path_value = sample.get(path_field)
    if path_value:
        rel = str(path_value)
        if local_index is not None:
            local_rel = _resolve_local_path(rel, local_index)
            if local_rel is not None:
                return local_rel, False
        repo_path = _resolve_repo_path(rel, repo_index)
        if repo_path is not None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                pass
            else:
                local_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename=repo_path,
                    repo_type="dataset",
                    cache_dir=str(raw_dir),
                )
                return Path(local_path), False
    # Fallback: map label -> folder and pick a deterministic file
    if label_name and local_index is not None:
        label_dir = _label_to_folder(label_name)
        candidates = local_index.get("labels", {}).get(label_dir, [])
        if candidates:
            pick = candidates[sample_index % len(candidates)]
            return pick, True
    return None, False


def _extract_pose_mediapipe(
    video_path: Path,
    frame_stride: int,
    max_frames: int,
    start_s: float | None,
    end_s: float | None,
):
    try:
        import cv2
    except ImportError as e:
        raise SystemExit("Missing dependency: opencv-python") from e

    try:
        import mediapipe as mp
    except ImportError as e:
        raise SystemExit("Missing dependency: mediapipe") from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    kpts = []
    conf = []
    t_list = []

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if start_s is not None and start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(start_s) * 1000.0)

    frame_idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if end_s is not None and t_ms > 0 and (t_ms / 1000.0) >= end_s:
            break
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue
        if kept >= max_frames:
            break
        frame_idx += 1
        kept += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks is None:
            kpts.append(np.zeros((17, 2), dtype=np.float32))
            conf.append(np.zeros((17,), dtype=np.float32))
        else:
            lm = res.pose_landmarks.landmark
            xy = np.zeros((17, 2), dtype=np.float32)
            cf = np.zeros((17,), dtype=np.float32)
            for j, idx in enumerate(MP33_TO_COCO17):
                xy[j, 0] = float(lm[idx].x)
                xy[j, 1] = float(lm[idx].y)
                cf[j] = float(lm[idx].visibility)
            kpts.append(xy)
            conf.append(cf)

        if t_ms and t_ms > 0:
            base = float(start_s) if start_s is not None else 0.0
            t_list.append(max(0.0, (t_ms / 1000.0) - base))
        else:
            t_list.append((kept - 1) / float(fps))

    cap.release()
    pose.close()

    if not kpts:
        return None

    return (
        np.asarray(t_list, dtype=np.float32),
        np.asarray(kpts, dtype=np.float32),
        np.asarray(conf, dtype=np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg, _project_root, raw_dir, pose_dir, manifest_path = _resolve_paths(config_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = str(cfg["omnifall"]["hf_dataset"])
    dataset_config = cfg["omnifall"].get("hf_config")
    split = args.split if args.split else str(cfg["omnifall"]["hf_split"])

    label_field = str(cfg["omnifall"]["label_field"])
    path_field = str(cfg["omnifall"].get("path_field", "path"))
    start_field = str(cfg["omnifall"].get("start_field", "start"))
    end_field = str(cfg["omnifall"].get("end_field", "end"))
    local_video_root = cfg["omnifall"].get("local_video_root")
    fall_values = set(str(v) for v in cfg["omnifall"]["fall_values"])
    non_fall_values = set(str(v) for v in cfg["omnifall"]["non_fall_values"])

    frame_stride = int(cfg["omnifall"]["frame_stride"])
    max_frames = int(cfg["omnifall"]["max_frames"])

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Missing dependency: datasets") from e

    ds = load_dataset(
        dataset_name,
        dataset_config if dataset_config not in (None, "null") else None,
        split=split,
        cache_dir=str(raw_dir),
    )

    label_names = None
    try:
        feat = ds.features.get(label_field)
        if hasattr(feat, "names"):
            label_names = list(feat.names)
    except Exception:
        label_names = None

    repo_index = _build_repo_index(dataset_name)
    local_index = None
    local_root = None
    if local_video_root:
        local_root = Path(local_video_root)
        if local_root.exists():
            local_index = _build_local_media_index(local_root)
    if local_index is None:
        extracted = _ensure_media_cache(dataset_name, raw_dir)
        if extracted is not None and extracted.exists():
            local_root = extracted
            local_index = _build_local_media_index(extracted)

    manifest_items = []
    total = len(ds)
    limit = args.limit if args.limit and args.limit > 0 else total

    skipped_label = 0
    skipped_video = 0

    for idx, sample in enumerate(tqdm(ds, total=min(limit, total))):
        if idx >= limit:
            break

        raw_label = sample.get(label_field)
        label_name = None
        if label_names is not None and isinstance(raw_label, (int, float)):
            raw_idx = int(raw_label)
            if 0 <= raw_idx < len(label_names):
                label_name = label_names[raw_idx]
        elif raw_label is not None:
            label_name = str(raw_label)

        label = _normalize_label(raw_label, fall_values, non_fall_values, label_names)
        if label is None:
            skipped_label += 1
            continue

        video_path, used_fallback = _get_video_path(
            sample,
            raw_dir,
            dataset_name,
            repo_index,
            path_field,
            local_index,
            local_root,
            label_name,
            idx,
        )
        if video_path is None or not video_path.exists():
            skipped_video += 1
            continue

        raw_id = sample.get("id")
        if raw_id is None:
            base = str(sample.get(path_field, "sample")).replace("/", "_")
            start_v = sample.get(start_field, 0.0)
            end_v = sample.get(end_field, 0.0)
            raw_id = f"{base}_{start_v:.2f}_{end_v:.2f}_{idx}"
        sample_id = str(raw_id)
        out_path = pose_dir / f"omnifall_{sample_id}.npz"
        if out_path.exists() and not args.overwrite:
            manifest_items.append({"id": sample_id, "pose_path": str(out_path), "label": int(label)})
            continue

        start_s = sample.get(start_field)
        end_s = sample.get(end_field)
        start_s = float(start_s) if start_s is not None else None
        end_s = float(end_s) if end_s is not None else None
        # If we had to fallback by label, ignore start/end clipping
        if used_fallback:
            start_s = None
            end_s = None

        res = _extract_pose_mediapipe(
            video_path,
            frame_stride=frame_stride,
            max_frames=max_frames,
            start_s=start_s,
            end_s=end_s,
        )
        if res is None:
            continue
        t_sec, kpts, conf = res

        np.savez_compressed(out_path, t=t_sec, kpts=kpts, conf=conf, label=np.int32(label))
        manifest_items.append({"id": sample_id, "pose_path": str(out_path), "label": int(label)})

    write_jsonl(manifest_path, manifest_items)
    print(f"Wrote OmniFall manifest: {manifest_path} (items={len(manifest_items)})")
    print(f"Skipped: label={skipped_label}, video={skipped_video}")


if __name__ == "__main__":
    main()
