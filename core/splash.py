"""Pre-import splash. Stdlib only — no Rich, no project deps.

`boot()` is called as the very first thing in `main.py`, before any heavy
import. It prints a prerendered ASCII-art "inst2vec" surrounded by blank
padding so the splash visually anchors the terminal during the 5–10 s
import phase that follows. Real output printed later scrolls the splash
naturally off the top of the viewport.
"""

from __future__ import annotations

import shutil
import sys

_LOGO_WIDTH = 76

# Prerendered "inst2vec" logo. Embedded as a literal so boot() has zero
# runtime dependencies and runs before any heavy import.
_LOGO_LINES: tuple[str, ...] = (
    "  ███                      █████     ████████",
    " ░░░                      ░░███     ███░░░░███",
    " ████  ████████    █████  ███████  ░░░    ░███ █████ █████  ██████   ██████",
    "░░███ ░░███░░███  ███░░  ░░░███░      ███████ ░░███ ░░███  ███░░███ ███░░███",
    " ░███  ░███ ░███ ░░█████   ░███      ███░░░░   ░███  ░███ ░███████ ░███ ░░░",
    " ░███  ░███ ░███  ░░░░███  ░███ ███ ███      █ ░░███ ███  ░███░░░  ░███  ███",
    " █████ ████ █████ ██████   ░░█████ ░██████████  ░░█████   ░░██████ ░░██████",
    "░░░░░ ░░░░ ░░░░░ ░░░░░░     ░░░░░  ░░░░░░░░░░    ░░░░░     ░░░░░░   ░░░░░░",
)


def boot() -> None:
    """Render the splash. No-op on non-TTY stdout."""
    if not sys.stdout.isatty():
        return

    cols, rows = shutil.get_terminal_size(fallback=(80, 24))

    art_lines = [line.ljust(_LOGO_WIDTH) for line in _LOGO_LINES]
    art_height = len(art_lines)

    top_pad = max((rows - art_height) // 2 - 1, 1)
    bottom_pad = max(rows - art_height - top_pad - 1, 1)

    out = "\n" * top_pad
    for line in art_lines:
        out += line.center(cols).rstrip() + "\n"
    out += "\n" * bottom_pad

    sys.stdout.write(out)
    sys.stdout.flush()
