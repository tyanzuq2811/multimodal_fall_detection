from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TrainLogger:
    def __init__(
        self,
        log_dir: Path,
        run_name: str,
        save_csv: bool = True,
        save_json: bool = True,
        wandb_cfg: dict[str, Any] | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.run_name = run_name
        self.save_csv = bool(save_csv)
        self.save_json = bool(save_json)
        self.csv_path = self.log_dir / f"{self.run_name}.csv"
        self.json_path = self.log_dir / f"{self.run_name}.json"
        self._csv_file = None
        self._csv_writer = None
        self._json_records: list[dict[str, Any]] = []
        self._wandb = None
        self._wandb_run = None

        self.log_dir.mkdir(parents=True, exist_ok=True)

        wandb_cfg = wandb_cfg or {}
        if wandb_cfg.get("enable"):
            try:
                import wandb  # type: ignore
            except Exception as exc:
                print(f"[WARN] wandb disabled: {exc}")
            else:
                mode = str(wandb_cfg.get("mode", "online"))
                project = wandb_cfg.get("project")
                entity = wandb_cfg.get("entity")
                tags = wandb_cfg.get("tags")
                name = wandb_cfg.get("run_name") or self.run_name
                self._wandb_run = wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    tags=tags,
                    mode=mode,
                )
                if run_config is not None:
                    self._wandb_run.config.update(run_config, allow_val_change=True)
                self._wandb = wandb

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        row = dict(metrics)
        row.setdefault("timestamp", datetime.utcnow().isoformat())

        if self.save_csv:
            if self._csv_writer is None:
                self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
                self._csv_writer.writeheader()
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        if self.save_json:
            self._json_records.append(row)
            with self.json_path.open("w", encoding="utf-8") as f:
                json.dump(self._json_records, f, ensure_ascii=True)

        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        if self._wandb_run is not None:
            self._wandb.finish()
