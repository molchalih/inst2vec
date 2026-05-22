"""Per-case visualization stage.

Mirror of modules/clustering/assign.py: per-case fingerprint gate over
upstream cluster_assign state and the UserCluster rows it produced.
Wipe-and-recompute on stale; commit a single transaction per case.

Fingerprint is deliberately data-only: config_hash is hash_text("")
because there are no settings knobs to track. dependency_hash chains
to Stage.CLUSTER_ASSIGN so the stage flags drift even when row content
happens to repeat.
"""

from __future__ import annotations

import time

from core import fingerprint as fp
from core.console import log
from core.database import (
    StageState,
    UserCluster,
    Visualization,
    VisualizationCluster,
    VisualizationUser,
    get_session,
)
from core.pipeline import Stage
from modules.embeddings.cases import CASE_REGISTRY
from modules.visualization.compute import build_case_payload

STAGE = Stage.VISUALIZATION


def _fingerprint(session, case: str) -> fp.Fingerprint:
    rows = (
        session.query(
            UserCluster.user_id,
            UserCluster.cluster_id,
            UserCluster.umap_x,
            UserCluster.umap_y,
        )
        .filter_by(embedding_case=case)
        .order_by(UserCluster.user_id)
        .all()
    )
    return fp.Fingerprint(
        data=fp.hash_rows(rows),
        config=fp.hash_text(""),
        dependency=fp.stage_dependency_hash(session, Stage.CLUSTER_ASSIGN, case),
    )


def _run_case(case: str) -> None:
    spec = CASE_REGISTRY.get(case)
    if spec is None:
        raise ValueError(f"unknown embedding case: {case}")
    scope = f"viz:{case}"
    t_stage = time.perf_counter()

    # 1. Gate on upstream + compute current fingerprint.
    session = get_session()
    try:
        upstream = session.get(StageState, ("cluster_assign", case))
        if upstream is None:
            log(scope, "SKIP", "cluster", "none")
            return
        current = _fingerprint(session, case)
        stale = fp.is_stale(session, STAGE, case, current)
        diff = fp.describe_diff(session, STAGE, case, current) if stale else ""
        user_rows = session.query(UserCluster).filter_by(embedding_case=case).all()
    finally:
        session.close()

    if not stale:
        log(scope, "SKIP", "fingerprint", "ok")
        return

    log(scope, "SCAN", "fingerprint", "stale", stats={"diff": diff})

    # 2. Compute in memory (no DB lock).
    payload = build_case_payload(case, spec.display_label, user_rows)

    # 3. Short write block.
    session = get_session()
    try:
        session.query(VisualizationUser).filter_by(embedding_case=case).delete()
        session.query(VisualizationCluster).filter_by(embedding_case=case).delete()
        session.merge(
            Visualization(
                embedding_case=case,
                label=spec.display_label,
                size=len(payload.users),
                source_hash=current.data,
            )
        )
        if payload.users:
            session.bulk_save_objects(payload.users)
        if payload.clusters:
            session.bulk_save_objects(payload.clusters)
        fp.mark_complete(session, STAGE, case, current)
        session.commit()
    finally:
        session.close()

    log(scope, "WRITE", "visualization_users", "ok", stats={"rows": len(payload.users)})
    log(
        scope,
        "WRITE",
        "visualization_clusters",
        "ok",
        stats={"rows": len(payload.clusters)},
    )
    log(
        scope,
        "SEAL",
        "visualization",
        "ok",
        stats={"time": time.perf_counter() - t_stage},
    )


def run_visualization(cases: tuple[str, ...]) -> None:
    """Per-case fingerprint-gated visualization build.

    Always processes every case in `cases` (including those with
    expose_to_viewer=False) so the DB always carries the latest rows;
    the JSON exporter, not this stage, decides what to expose.
    """
    for case in cases:
        _run_case(case)
