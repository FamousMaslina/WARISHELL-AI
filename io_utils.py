from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunIO:
    def __init__(self, out_dir: Path) -> None:
        self.root = out_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def turn_dir(self, t: int) -> Path:
        p = self.root / f"turn_{t:03d}"
        p.mkdir(exist_ok=True)
        return p

    def save_json(self, path: Path, data: Any) -> None:
        text = json.dumps(data, indent=2, ensure_ascii=False)
        path.write_text(text, encoding="utf-8", newline="\n")

    def save_text(self, path: Path, data: str) -> None:
        path.write_text(data, encoding="utf-8", newline="\n")
