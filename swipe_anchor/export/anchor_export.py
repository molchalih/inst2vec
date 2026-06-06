"""Build the Stage-3 hand-off artifact (plan §1.3).

Writes a read-only ``anchor_export/`` bundle from the app store — the only thing
the pipeline consumes from this app (no app->pipeline DB coupling):

    triplets.jsonl        one triplet per line, with seed-group / modality /
                          weight / agreement provenance
    embedding.npy         learned target geometry [n_users, d]  (only when a
                          geometry is supplied by the live worker; absent at MVP)
    embedding_index.json  row i -> user_id  (for embedding.npy)
    meta.json             schema version, counts, build timestamp, export hash

``user_id`` is always the pipeline's anonymous ``users.id`` — never a username.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from swipe_anchor.db.models import Comparison, Consensus, Triplet

SCHEMA_VERSION = 1


def _triplet_rows(session: Session) -> list[dict]:
    stmt = (
        select(Triplet, Comparison, Consensus)
        .join(Comparison, Triplet.comparison_id == Comparison.comparison_id)
        .join(
            Consensus,
            Consensus.comparison_id == Triplet.comparison_id,
            isouter=True,
        )
        .order_by(Triplet.id)
    )
    rows: list[dict] = []
    for triplet, comparison, consensus in session.execute(stmt):
        resolved = bool(consensus.resolved) if consensus is not None else False
        rows.append(
            {
                "anchor_user_id": triplet.anchor_id,
                "positive_user_id": triplet.positive_id,
                "negative_user_id": triplet.negative_id,
                "seed_group": comparison.seed_group,
                "expected_modality": comparison.expected_modality,
                "weight": triplet.weight,
                "n_judgments": comparison.n_judgments,
                "agreement": consensus.agreement if consensus is not None else None,
                "source": "consensus" if resolved else "raw",
            }
        )
    return rows


def export_anchor(
    session: Session,
    out_dir: str | Path,
    *,
    build_timestamp: datetime,
    geometry: dict[int, np.ndarray] | None = None,
) -> dict:
    """Write the export bundle to ``out_dir`` and return the manifest (meta.json).

    ``build_timestamp`` is injected (never read from the clock here) so exports
    are reproducible and testable. ``geometry`` maps ``user_id -> vector``; when
    omitted the embedding files are skipped (the MVP has no learned geometry yet).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = _triplet_rows(session)
    triplets_path = out / "triplets.jsonl"
    with triplets_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")

    n_users = 0
    if geometry:
        user_ids = sorted(geometry)
        matrix = np.vstack(
            [np.asarray(geometry[u], dtype=np.float32) for u in user_ids]
        )
        np.save(out / "embedding.npy", matrix)
        (out / "embedding_index.json").write_text(
            json.dumps({str(i): u for i, u in enumerate(user_ids)}, sort_keys=True)
        )
        n_users = len(user_ids)

    export_hash = hashlib.sha256(triplets_path.read_bytes()).hexdigest()
    meta = {
        "schema_version": SCHEMA_VERSION,
        "build_timestamp": build_timestamp.isoformat(),
        "export_hash": export_hash,
        "counts": {"triplets": len(rows), "users": n_users},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta
