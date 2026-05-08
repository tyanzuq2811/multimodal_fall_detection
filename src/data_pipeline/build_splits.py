from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.config import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_config.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    project_root = config_path.resolve().parent.parent
    splits_dir = project_root / Path(cfg["paths"]["splits_dir"])
    out_path = (
        Path(args.out)
        if args.out
        else splits_dir / "upfall_subject_split.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split_cfg = cfg.get("split", {})
    payload = {
        "train": split_cfg.get("train_subjects", []),
        "val": split_cfg.get("val_subjects", []),
        "test": split_cfg.get("test_subjects", []),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Wrote subject split: {out_path}")


if __name__ == "__main__":
    main()
