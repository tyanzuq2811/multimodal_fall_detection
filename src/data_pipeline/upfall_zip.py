from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


_FRAME_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2}\.\d+)$")


def frame_name_to_iso_ts(frame_name: str) -> str:
    # Example zip member name: 2018-07-04T12_04_17.738369.png
    stem = Path(frame_name).name
    stem = stem.rsplit(".", 1)[0]
    m = _FRAME_TS_RE.match(stem)
    if not m:
        # fallback: replace underscores in time part
        if "T" in stem:
            date, time = stem.split("T", 1)
            return f"{date}T{time.replace('_', ':')}"
        return stem
    date, hh, mm, rest = m.group(1), m.group(2), m.group(3), m.group(4)
    # rest is like 17.738369
    return f"{date}T{hh}:{mm}:{rest}"


def parse_frame_timestamp(frame_name: str) -> datetime:
    iso_str = frame_name_to_iso_ts(frame_name)
    # Handle microseconds > 6 digits (truncate to 6 digits for fromisoformat)
    # e.g., 2018-07-10T13:27:18.0326237 -> 2018-07-10T13:27:18.032623
    if '.' in iso_str:
        parts = iso_str.rsplit('.', 1)
        if len(parts[1]) > 6:
            iso_str = f"{parts[0]}.{parts[1][:6]}"
    return datetime.fromisoformat(iso_str)


@dataclass(frozen=True)
class ZipFrames:
    zip_path: Path
    members: list[str]
    timestamps: list[datetime]


def list_frames(zip_path: str | Path) -> ZipFrames:
    zp = Path(zip_path)
    with zipfile.ZipFile(zp, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".png", ".jpg", ".jpeg"))]

    members.sort()
    timestamps = [parse_frame_timestamp(m) for m in members]
    return ZipFrames(zip_path=zp, members=members, timestamps=timestamps)
