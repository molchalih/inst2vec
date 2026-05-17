"""User-level embedding aggregation.

Stage is wired to the fingerprint layer (modules.fingerprint). For each
embedding_case the stage:

  1. computes Fingerprint(data, config, dependency) from the actual
     ClipEmbedding rows for the case;
  2. on fingerprint match, logs and skips;
  3. on config drift, wipes every UserEmbedding row for the case so the
     incremental diff that follows cannot trust any stored hash;
  4. otherwise diffs per-user source hashes against stored ones,
     recomputes only the changed users, deletes orphan users, writes
     source_hash on every merged row, and seals the stage.

config_hash is currently ``"agg=mean_pool|v=1"``. Bump the version tag
when the aggregator changes.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from core import fingerprint as fp
from core.console import log
from core.database import (
    StageState,
    UserEmbedding,
    get_session,
)
from modules.embeddings.cases import DEFAULT_CASES
from modules.embeddings.state import (
    get_clip_embedding_rows_for_user_aggregation,
    get_stored_user_hashes,
    per_user_source_hashes,
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


def _wipe_case(session, case: str) -> None:
    session.query(UserEmbedding).filter_by(embedding_case=case).delete()
    session.commit()


def _delete_users(session, case: str, user_ids: set[int]) -> None:
    if not user_ids:
        return
    session.query(UserEmbedding).filter(
        UserEmbedding.embedding_case == case,
        UserEmbedding.user_id.in_(user_ids),
    ).delete(synchronize_session=False)
    session.commit()


def _recompute_users(
    session,
    case: str,
    rows: list[tuple[int, bytes, int]],
    user_ids: set[int],
    desired_hashes: dict[int, str],
) -> None:
    if not user_ids:
        return
    subset = [r for r in rows if r[2] in user_ids]
    aggregated = aggregate_user_embeddings_from_rows(subset)
    log(f"embed:user:{case}", f"{len(aggregated)} users to (re-)embed")
    for user_id, mean_blob in aggregated.items():
        session.merge(
            UserEmbedding(
                user_id=user_id,
                embedding_case=case,
                embedding=mean_blob,
                source_hash=desired_hashes[user_id],
            )
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

            stored_state = session.get(StageState, (STAGE, case))
            if stored_state is not None and stored_state.config_hash != current.config:
                log(
                    f"embed:user:{case}",
                    "config drift — wiping case before incremental diff",
                )
                _wipe_case(session, case)

            desired = per_user_source_hashes(rows)
            stored = get_stored_user_hashes(session, case)
            changed = fp.row_diff(desired, stored)
            orphans = set(stored) - set(desired)

            _delete_users(session, case, orphans)
            _recompute_users(session, case, rows, changed, desired)

            fp.mark_complete(session, STAGE, case, current)
            session.commit()
            log(f"embed:user:{case}", "done", level="ok")
    finally:
        session.close()
