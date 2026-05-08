from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("Each JSONL line must be an object")
            items.append(obj)
    return items


def write_jsonl(path: str | Path, items: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
