"""User-level embedding aggregation.

Stage is wired to the fingerprint layer (modules.fingerprint). For each
embedding_case the stage:

  1. computes Fingerprint(data, config, dependency) from the actual
     ClipEmbedding rows for the case;
  2. if stale, deletes its UserEmbedding rows for the case, recomputes,
     and merges StageState; commits once at the end so the seal lands
     in the same transaction as the rewrite;
  3. if not stale, logs and skips.

config_hash is currently constant ("agg=mean_pool|v=1"). Bump the
version tag when the aggregator changes.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from modules import fingerprint as fp
from modules.console import log
from modules.database import (
    Base,
    UserEmbedding,
    get_engine,
    get_session,
)
from modules.embeddings.cases import DEFAULT_CASES
from modules.embeddings.state import (
    get_clip_embedding_rows_for_user_aggregation,
)
from modules.embeddings.vectors import bytes_to_array

STAGE = "user_embeddings"
_CONFIG_IDENTITY = "agg=mean_pool|v=1"


def aggregate_user_embeddings_from_rows(
    rows: list[tuple[int, bytes, int]],
) -> dict[int, bytes]:
    """Mean-pool clip embedding blobs by user. Returns {user_id: mean_blob}.

    Accepts ``(clip_id, blob, user_id)`` triples (clip_id ignored here)
    so the fingerprint and aggregation share one source of truth.
    """
    user_arrays: dict[int, list[np.ndarray]] = defaultdict(list)
    for _clip_id, blob, user_id in rows:
        user_arrays[user_id].append(bytes_to_array(blob))
    return {
        user_id: np.stack(arrays).mean(axis=0).astype(np.float32).tobytes()
        for user_id, arrays in user_arrays.items()
    }


def _compute_fingerprint(
    session, case: str, rows: list[tuple[int, bytes, int]]
) -> fp.Fingerprint:
    # Hash the same filtered rows ``_recompute_case`` averages so the
    # fingerprint flips whenever aggregation membership or content does
    # (e.g. a clip deselected, a user marked ineligible, or — same row,
    # new bytes within one SQLite second).
    dep = fp.hash_rows(
        (clip_id, hashlib.sha256(blob).hexdigest()) for clip_id, blob, _ in rows
    )
    user_ids = sorted({user_id for _, _, user_id in rows})
    data = fp.hash_rows((uid,) for uid in user_ids)
    return fp.Fingerprint(
        data=data,
        config=fp.hash_text(_CONFIG_IDENTITY),
        dependency=dep,
    )


def _clear_case(session, case: str) -> None:
    session.query(UserEmbedding).filter_by(embedding_case=case).delete()
    session.commit()


def _recompute_case(session, case: str, rows: list[tuple[int, bytes, int]]) -> None:
    aggregated = aggregate_user_embeddings_from_rows(rows)
    log(f"embed:user:{case}", f"{len(aggregated)} users to embed")
    for user_id, mean_blob in aggregated.items():
        session.merge(
            UserEmbedding(user_id=user_id, embedding_case=case, embedding=mean_blob)
        )
        session.commit()


def embed_user_embeddings(settings, cases: list[str] | None = None) -> None:
    """Recompute and merge UserEmbedding rows for each case when stale.

    Reads ``settings.embeddings.exclude_disqualified_users`` so the
    aggregation and its fingerprint mirror the clip-embedding stage's
    candidate filter — preventing ineligible-user clip embeddings from
    flowing into clustering.
    """
    case_names = list(cases) if cases is not None else list(DEFAULT_CASES)
    exclude_disqualified = settings.embeddings.exclude_disqualified_users
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for case in case_names:
            rows = get_clip_embedding_rows_for_user_aggregation(
                session, case, exclude_disqualified
            )
            current = _compute_fingerprint(session, case, rows)
            if not fp.is_stale(session, STAGE, case, current):
                log(f"embed:user:{case}", "fingerprint match — skipping")
                continue

            diff = fp.describe_diff(session, STAGE, case, current)
            log(f"embed:user:{case}", f"stale ({diff}) — recomputing")
            _clear_case(session, case)
            _recompute_case(session, case, rows)
            fp.mark_complete(session, STAGE, case, current)
            session.commit()
            log(f"embed:user:{case}", "done", level="ok")
    finally:
        session.close()
