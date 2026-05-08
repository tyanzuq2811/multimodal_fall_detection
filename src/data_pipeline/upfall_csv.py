from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


SPECIAL_GROUP_COLUMNS = {
    "TimeStamps",
    "Subject",
    "Activity",
    "Trial",
    "BrainSensor",
    "Infrared1",
    "Infrared2",
    "Infrared3",
    "Infrared4",
    "Infrared5",
    "Infrared6",
}


def _axis_name(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s.startswith("x-axis"):
        return "x"
    if s.startswith("y-axis"):
        return "y"
    if s.startswith("z-axis"):
        return "z"
    if s == "":
        return ""
    out = "".join(ch if ch.isalnum() else "_" for ch in s)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _forward_fill(values: list[str]) -> list[str]:
    cur = ""
    out: list[str] = []
    for v in values:
        if v != "":
            cur = v
        out.append(cur)
    return out


def _infer_column_names(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header_row1 = next(reader)
        header_row2 = next(reader)
        first_data_row = next(reader)

    n_cols = max(len(header_row1), len(header_row2), len(first_data_row))
    header_row1 = header_row1 + [""] * (n_cols - len(header_row1))
    header_row2 = header_row2 + [""] * (n_cols - len(header_row2))

    groups = _forward_fill([c.strip() for c in header_row1])
    axes = [_axis_name(c) for c in header_row2]

    names: list[str] = []
    seen: dict[str, int] = {}
    for group, axis in zip(groups, axes):
        if group == "":
            base = "col"
        elif group in SPECIAL_GROUP_COLUMNS:
            base = group
        else:
            base = group if axis == "" else f"{group}_{axis}"

        count = seen.get(base, 0)
        if count:
            base = f"{base}_{count + 1}"
        seen[base] = count + 1
        names.append(base)
    return names


def read_upfall_csv(csv_path: str | Path) -> pd.DataFrame:
    p = Path(csv_path)
    col_names = _infer_column_names(p)
    # index_col=False prevents pandas from shifting the first field into the index
    # when the file has one extra trailing comma/column.
    df = pd.read_csv(p, names=col_names, skiprows=2, engine="python", index_col=False)
    return df


def parse_upfall_timestamp(ts: str) -> datetime:
    # Example: 2018-07-04T12:04:17.738369
    return datetime.fromisoformat(ts)


def timestamps_to_seconds(timestamps: Iterable[str]) -> tuple[list[datetime], list[float]]:
    dts: list[datetime] = []
    for ts in timestamps:
        dts.append(parse_upfall_timestamp(str(ts)))
    if not dts:
        return [], []
    t0 = dts[0]
    secs = [(dt - t0).total_seconds() for dt in dts]
    return dts, secs


@dataclass(frozen=True)
class TrialInfo:
    subject: int
    activity: int
    trial: int


def get_trial_info(df: pd.DataFrame) -> TrialInfo:
    if df.empty:
        raise ValueError("Empty UP-Fall CSV")
    # Columns are constants per trial.
    subject = int(df["Subject"].iloc[0])
    activity = int(df["Activity"].iloc[0])
    trial = int(df["Trial"].iloc[0])
    return TrialInfo(subject=subject, activity=activity, trial=trial)


def extract_accel_6d(df: pd.DataFrame, wrist_group: str = "WristAccelerometer", belt_group: str = "BeltAccelerometer"):
    required = [
        f"{wrist_group}_x",
        f"{wrist_group}_y",
        f"{wrist_group}_z",
        f"{belt_group}_x",
        f"{belt_group}_y",
        f"{belt_group}_z",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing accelerometer columns: {missing}")

    ts = df["TimeStamps"].astype(str).tolist()
    _, t_sec = timestamps_to_seconds(ts)
    values = df[required].astype(float).to_numpy()
    return t_sec, values
