#!/usr/bin/env python
"""Copy the visualization JSON export into frontend/public/data/.

Deliberate release-time step. After running the pipeline locally so the
visualization stage writes ``data/visualization/manifest.json`` and
``data/visualization/runs/<case>/{users,clusters}.json``, invoke this
script to stage that snapshot into the tracked frontend public-assets
directory. The Pages workflow picks it up via `vite build`.

Idempotent: re-running produces an identical destination tree.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv()

from core.config import load_runtime_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISH_DEST = REPO_ROOT / "frontend" / "public" / "data"


def publish(source: Path, dest: Path) -> dict:
    """Copy ``source`` into ``dest`` after wiping ``dest``.

    Returns the parsed manifest so callers can print a release summary.
    Raises ``FileNotFoundError`` if ``source`` or ``source/manifest.json``
    is missing.
    """
    if not source.exists():
        raise FileNotFoundError(
            f"visualization export not found at {source}; "
            "run `uv run python main.py` first"
        )
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing manifest.json under {source}; nothing to publish"
        )
    manifest = json.loads(manifest_path.read_text())

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)
    return manifest


def _format_summary(manifest: dict, dest: Path) -> str:
    total_bytes = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    runs = manifest.get("runs", [])
    rel = dest.relative_to(REPO_ROOT) if dest.is_relative_to(REPO_ROOT) else dest
    lines = [
        f"published {len(runs)} run(s) to {rel}",
        f"total size: {total_bytes / 1024:.1f} KiB",
    ]
    for run in runs:
        rid = run.get("id")
        case = run.get("case")
        label = run.get("label")
        size = run.get("size")
        lines.append(f"  - {rid} (case={case}, label={label!r}, size={size})")
    return "\n".join(lines)


def main() -> int:
    settings, _ = load_runtime_config()
    source = Path(settings.visualization.export_dir).resolve()
    manifest = publish(source, PUBLISH_DEST)
    print(_format_summary(manifest, PUBLISH_DEST))
    return 0


if __name__ == "__main__":
    sys.exit(main())
