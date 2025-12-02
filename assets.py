from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from config import FLAGS_DIR_CANDIDATES
from models import Agent


def _resolve_flag_path(name_or_path: Optional[str]) -> Optional[Path]:
    if not name_or_path:
        return None
    p = Path(name_or_path)
    if p.is_file():
        return p
    for base in FLAGS_DIR_CANDIDATES:
        q = base / name_or_path
        if q.is_file():
            return q
    return None


def ensure_flag_assets(out_dir: Path, agents: List[Agent]) -> None:
    """
    Copy each agent's flag file (if found) into {out_dir}/flags/, preserving filename.
    viewer.html can then use 'flags/<filename>' relative to the run root.
    """
    dest = out_dir / "flags"
    dest.mkdir(parents=True, exist_ok=True)
    for a in agents:
        if not a.flag:
            continue
        src = _resolve_flag_path(a.flag)
        if not src:
            continue
        try:
            shutil.copy2(src, dest / src.name)
        except Exception:
            pass
