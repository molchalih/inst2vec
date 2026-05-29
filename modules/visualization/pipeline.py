"""Per-case visualization stage.

Mirror of modules/clustering/assign.py: per-case fingerprint gate over
upstream cluster_assign state and the UserCluster rows it produced.
Wipe-and-recompute on stale; commit a single transaction per case.

Fingerprint is deliberately data-only: config_hash is hash_text("")
because there are no settings knobs to track. dependency_hash chains
to Stage.CLUSTER_ASSIGN (layout/rows) and the cluster_labels stage (the
embedded per-cluster label block) so the stage flags drift even when the
UserCluster rows happen to repeat — e.g. after a pure re-labelling run.
"""

from __future__ import annotations

import time

from core import fingerprint as fp
from core.database import (
    StageState,
    UserCluster,
    Visualization,
    VisualizationCluster,
    VisualizationUser,
    get_session,
)
from core.log import event, scope
from core.pipeline import Stage
from modules.embeddings.cases import CASE_REGISTRY
from modules.labels.state import STAGE_CLUSTER_LABELS, cluster_scope_for
from modules.visualization.compute import build_case_payload

STAGE = Stage.VISUALIZATION


def _fingerprint(session, case: str) -> fp.Fingerprint:
    rows = (
        session.query(
            UserCluster.user_id,
            UserCluster.cluster_id,
            UserCluster.umap_x,
            UserCluster.umap_y,
            UserCluster.centrality,
        )
        .filter_by(embedding_case=case)
        .order_by(UserCluster.user_id)
        .all()
    )
    return fp.Fingerprint(
        data=fp.hash_rows(rows),
        config=fp.hash_text(""),
        # Chain to CLUSTER_ASSIGN (the UMAP layout / cluster rows) AND
        # CLUSTER_LABELS (the per-cluster label block the atlas embeds), so
        # re-labelling drifts this stage and forces a fresh export instead of
        # publishing a stale label.
        dependency=fp.compose_hashes(
            fp.stage_dependency_hash(session, Stage.CLUSTER_ASSIGN, case),
            fp.stage_dependency_hash(
                session, STAGE_CLUSTER_LABELS, cluster_scope_for(case)
            ),
        ),
    )


@scope("visualization:{case}")
def _run_case(case: str) -> None:
    spec = CASE_REGISTRY.get(case)
    if spec is None:
        raise ValueError(f"unknown embedding case: {case}")
    t_stage = time.perf_counter()

    # 1. Gate on upstream + compute current fingerprint.
    session = get_session()
    try:
        upstream = session.get(StageState, ("cluster_assign", case))
        if upstream is None:
            event("SKIP", "cluster")
            return
        current = _fingerprint(session, case)
        stale = fp.is_stale(session, STAGE, case, current)
        diff = fp.describe_diff(session, STAGE, case, current) if stale else ""
        user_rows = session.query(UserCluster).filter_by(embedding_case=case).all()
    finally:
        session.close()

    if not stale:
        event("SKIP", "fingerprint")
        return

    event("SCAN", "fingerprint", stats={"diff": diff})

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

    event("WRITE", "visualization_users", stats={"rows": len(payload.users)})
    event("WRITE", "visualization_clusters", stats={"rows": len(payload.clusters)})
    event("SEAL", "visualization", stats={"time": time.perf_counter() - t_stage})


@scope("visualization")
def run_visualization(cases: tuple[str, ...]) -> None:
    """Per-case fingerprint-gated visualization build.

    Always processes every case in `cases` (including those with
    expose_to_viewer=False) so the DB always carries the latest rows;
    the JSON exporter, not this stage, decides what to expose.

    The stage SEAL is emitted by the public `modules.visualization.run`
    wrapper so a failure in `export_visualization_json` is reported as
    a failed Visualization stage.
    """
    for case in cases:
        _run_case(case)
