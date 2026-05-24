"""Atomic JSON writer for the atlas export contract.

Both Phase 1's synthetic fixture generator (`scripts/gen_atlas_fixture.py`)
and Phase 3's real DB-backed exporter (`scripts/export_atlas.py`) call
into this module. There is exactly one writer for these JSON files —
the SSOT for the contract.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .schemas import ClustersFile, Manifest, ManifestRun, UsersFile


@dataclass
class RunPayload:
    """One run, ready to write. The exporter and the fixture generator
    both produce this shape; the writer doesn't care which."""

    meta: ManifestRun
    users: UsersFile
    clusters: ClustersFile


def _dump_json(obj: object) -> str:
    """Stable JSON: sorted keys, fixed separators, trailing newline.

    Idempotence depends on this — re-running the writer with the same
    input must produce byte-identical files.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"


def _write_atomic(path: Path, payload: str) -> None:
    """Write via temp-file + rename so the frontend never reads a
    half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def write_manifest(out_dir: Path, manifest: Manifest) -> Path:
    path = out_dir / "manifest.json"
    _write_atomic(path, _dump_json(manifest.model_dump()))
    return path


def write_run(out_dir: Path, run: RunPayload) -> tuple[Path, Path]:
    if run.users.run_id != run.meta.id or run.clusters.run_id != run.meta.id:
        raise ValueError(
            f"run_id mismatch: meta.id={run.meta.id!r}, "
            f"users.run_id={run.users.run_id!r}, "
            f"clusters.run_id={run.clusters.run_id!r}"
        )
    run_dir = out_dir / "runs" / run.meta.id
    users_path = run_dir / "users.json"
    clusters_path = run_dir / "clusters.json"
    _write_atomic(users_path, _dump_json(run.users.model_dump()))
    _write_atomic(clusters_path, _dump_json(run.clusters.model_dump()))
    return users_path, clusters_path


def write_dataset(
    out_dir: Path,
    *,
    default_run_id: str,
    runs: Sequence[RunPayload],
) -> None:
    """Write the manifest plus every run's payload in one call."""
    if not runs:
        raise ValueError("write_dataset requires at least one run")
    manifest = Manifest(
        default_run_id=default_run_id,
        runs=[r.meta for r in runs],
    )
    write_manifest(out_dir, manifest)
    for run in runs:
        write_run(out_dir, run)
