"""Orphan GC for the labels stage.

Stage-1 / stage-2 fingerprints are config + dep only with respect to
the set of selected clips / live clusters: adding a row drifts the
data hash (clip-pass ``_data_hash_for_video`` / cluster-pass candidate
payloads), but *removing* one from the source set does not. Without an explicit
purge, dropping a user from ``data.csv`` or unselecting a clip leaves
inert ``clip_labels`` / ``cluster_labels`` rows behind forever.

``purge_orphans`` runs as the first thing ``pipeline.run`` does, before
any fingerprint compute. Caller owns the commit.
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core.database import (
    Clip,
    ClipLabel,
    ClusterLabel,
    UserCluster,
    clip_used_in_analysis,
)


def purge_orphans(session: Session) -> tuple[int, int]:
    """Delete dead clip_labels / cluster_labels rows.

    Caller owns the transaction (no internal commit). Returns
    ``(clip_rows_deleted, cluster_rows_deleted)``. Idempotent:
    re-running on a clean DB returns ``(0, 0)``.
    """
    eligible_ids = set(
        session.execute(select(Clip.id).where(*clip_used_in_analysis())).scalars().all()
    )
    if eligible_ids:
        clip_result = session.execute(
            delete(ClipLabel).where(ClipLabel.clip_id.notin_(eligible_ids))
        )
    else:
        clip_result = session.execute(delete(ClipLabel))

    live_cluster_keys = set(
        session.execute(
            select(UserCluster.embedding_case, UserCluster.cluster_id).where(
                UserCluster.cluster_id >= 0
            )
        ).all()
    )
    all_cluster_keys = set(
        session.execute(
            select(ClusterLabel.embedding_case, ClusterLabel.cluster_id)
        ).all()
    )
    stale_cluster_keys = all_cluster_keys - live_cluster_keys
    cluster_rows_deleted = 0
    for case, cid in stale_cluster_keys:
        result = session.execute(
            delete(ClusterLabel).where(
                ClusterLabel.embedding_case == case,
                ClusterLabel.cluster_id == cid,
            )
        )
        cluster_rows_deleted += int(result.rowcount or 0)

    return (int(clip_result.rowcount or 0), cluster_rows_deleted)
