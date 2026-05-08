from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UpfallTrialPaths:
    subject: int
    activity: int
    trial: int
    trial_dir: Path
    csv_path: Path
    camera_zips: dict[int, Path]


def iter_upfall_trials(raw_upfall_dir: str | Path) -> list[UpfallTrialPaths]:
    root = Path(raw_upfall_dir)
    trials: list[UpfallTrialPaths] = []
    if not root.exists():
        return trials

    for subject_dir in sorted(root.glob("Subject*")):
        if not subject_dir.is_dir():
            continue
        try:
            subject = int(subject_dir.name.replace("Subject", ""))
        except ValueError:
            continue
        for activity_dir in sorted(subject_dir.glob("Activity*")):
            if not activity_dir.is_dir():
                continue
            try:
                activity = int(activity_dir.name.replace("Activity", ""))
            except ValueError:
                continue
            for trial_dir in sorted(activity_dir.glob("Trial*")):
                if not trial_dir.is_dir():
                    continue
                try:
                    trial = int(trial_dir.name.replace("Trial", ""))
                except ValueError:
                    continue

                csv_candidates = list(trial_dir.glob("*.csv"))
                if not csv_candidates:
                    continue
                # Expect exactly one CSV per trial
                csv_path = sorted(csv_candidates)[0]

                camera_zips: dict[int, Path] = {}
                for zip_path in trial_dir.glob("*Camera*.zip"):
                    name = zip_path.name
                    if "Camera1" in name:
                        camera_zips[1] = zip_path
                    elif "Camera2" in name:
                        camera_zips[2] = zip_path

                trials.append(
                    UpfallTrialPaths(
                        subject=subject,
                        activity=activity,
                        trial=trial,
                        trial_dir=trial_dir,
                        csv_path=csv_path,
                        camera_zips=camera_zips,
                    )
                )

    return trials
