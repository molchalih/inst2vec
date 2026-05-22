"""JSON exporter for the visualization stage.

Reads the three visualization tables and writes the bulk-JSON tree the
frontend expects (manifest.json + runs/{case}/{users,clusters}.json).
Filters out cases whose EmbeddingCaseSpec.expose_to_viewer is False so
hidden cases never leak into the user-facing artifact.

Stale run directories and manifests from previous exports are pruned so
the on-disk tree always reflects the current DB / case set.

No fingerprint: always runs after the DB-write stage. A schema bump
(see schema.py SCHEMA_VERSION) thus rewrites every file on the next
pipeline call with no DB change.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from core.console import log
from core.database import (
    Visualization,
    VisualizationCluster,
    VisualizationUser,
    get_session,
)
from modules.embeddings.cases import CASE_REGISTRY
from modules.visualization.schema import SCHEMA_VERSION


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _bounds(users: list[VisualizationUser]) -> dict[str, float]:
    if not users:
        return {"minX": -1.0, "maxX": 1.0, "minY": -1.0, "maxY": 1.0}
    xs = [u.x for u in users]
    ys = [u.y for u in users]
    return {
        "minX": float(min(xs)),
        "maxX": float(max(xs)),
        "minY": float(min(ys)),
        "maxY": float(max(ys)),
    }


def _exposed_cases(cases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        c for c in cases if c in CASE_REGISTRY and CASE_REGISTRY[c].expose_to_viewer
    )


def _prune_stale_run_dirs(runs_dir: Path, keep: set[str]) -> None:
    if not runs_dir.exists():
        return
    for child in runs_dir.iterdir():
        if child.is_dir() and child.name not in keep:
            shutil.rmtree(child)


def export_visualization_json(settings, cases: tuple[str, ...]) -> None:
    """Read DB → write the frontend bulk-JSON tree. Idempotent."""
    viz_settings = settings.visualization
    export_dir = Path(viz_settings.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = export_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    manifest_path = export_dir / "manifest.json"

    cases_to_export = _exposed_cases(cases)
    session = get_session()
    try:
        viz_rows = {
            row.embedding_case: row
            for row in session.query(Visualization)
            .filter(Visualization.embedding_case.in_(cases_to_export))
            .all()
        }
        manifest_runs: list[dict] = []
        written_cases: set[str] = set()
        for case in cases_to_export:
            viz = viz_rows.get(case)
            if viz is None:
                log("viz:export", "SKIP", case, "no data")
                continue
            users = (
                session.query(VisualizationUser)
                .filter_by(embedding_case=case)
                .order_by(VisualizationUser.user_id)
                .all()
            )
            clusters = (
                session.query(VisualizationCluster)
                .filter_by(embedding_case=case)
                .order_by(VisualizationCluster.cluster_id)
                .all()
            )
            _write_json(
                export_dir / "runs" / case / "users.json",
                {
                    "version": SCHEMA_VERSION,
                    "run_id": case,
                    "bounds": _bounds(users),
                    "users": [[u.user_id, u.x, u.y, u.cluster_id] for u in users],
                },
            )
            _write_json(
                export_dir / "runs" / case / "clusters.json",
                {
                    "version": SCHEMA_VERSION,
                    "run_id": case,
                    "clusters": [
                        {
                            "id": c.cluster_id,
                            "label": c.label,
                            "cx": c.cx,
                            "cy": c.cy,
                            "rx": c.rx,
                            "ry": c.ry,
                            "angle": c.angle,
                            "size": c.size,
                        }
                        for c in clusters
                    ],
                },
            )
            manifest_runs.append(
                {
                    "id": case,
                    "case": case,
                    "label": viz.label,
                    "size": viz.size,
                }
            )
            written_cases.add(case)
            log(
                "viz:export",
                "WRITE",
                case,
                "ok",
                stats={"users": len(users), "clusters": len(clusters)},
            )

        _prune_stale_run_dirs(runs_dir, written_cases)

        if not manifest_runs:
            if manifest_path.exists():
                manifest_path.unlink()
            log("viz:export", "SKIP", "manifest", "no runs")
            return

        _write_json(
            manifest_path,
            {
                "version": SCHEMA_VERSION,
                "default_run_id": viz_settings.default_case,
                "runs": manifest_runs,
            },
        )
        log("viz:export", "SEAL", "manifest", "ok", stats={"runs": len(manifest_runs)})
    finally:
        session.close()
