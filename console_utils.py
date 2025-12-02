from __future__ import annotations

from pathlib import Path
from rich.console import Console

# Single shared console instance for the whole app
console = Console()
_TEE_ACTIVE = False
_TEE_STATE = {}


def setup_console_tee(out_dir: Path) -> None:
    """
    Mirror every console line to runs/.../console.log in real time.
    Safe to call multiple times; no-ops after first.
    """
    global _TEE_ACTIVE, _TEE_STATE
    if _TEE_ACTIVE:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "console.log"
    fh = open(log_path, "a", encoding="utf-8", newline="\n")

    file_console = Console(file=fh, color_system=None, force_terminal=False, no_color=True)
    orig_print = console.print
    orig_rule = console.rule

    def tee_print(*args, **kwargs):
        orig_print(*args, **kwargs)
        try:
            file_console.print(*args, **kwargs)
            fh.flush()
        except Exception:
            pass

    def tee_rule(*args, **kwargs):
        orig_rule(*args, **kwargs)
        try:
            file_console.rule(*args, **kwargs)
            fh.flush()
        except Exception:
            pass

    console.print = tee_print  # type: ignore[attr-defined]
    console.rule = tee_rule  # type: ignore[attr-defined]
    _TEE_STATE = {"fh": fh, "orig_print": orig_print, "orig_rule": orig_rule}
    _TEE_ACTIVE = True
