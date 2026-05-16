"""User-level embedding aggregation.

The public ``embed_user_embeddings(settings, cases=None)`` entry point is
added in a later task once state and case-registry submodules exist.

TODO: when the universal idempotence/hash module lands, guard the
recompute-and-merge cycle with a recipe-hash check so this stage becomes
idempotent without touching every clip embedding row.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from modules.embeddings.vectors import bytes_to_array


def aggregate_user_embeddings_from_rows(
    rows: list[tuple[bytes, int]],
) -> dict[int, bytes]:
    """Mean-pool clip embedding blobs by user. Returns {user_id: mean_blob}."""
    user_arrays: dict[int, list[np.ndarray]] = defaultdict(list)
    for blob, user_id in rows:
        user_arrays[user_id].append(bytes_to_array(blob))
    return {
        user_id: np.stack(arrays).mean(axis=0).astype(np.float32).tobytes()
        for user_id, arrays in user_arrays.items()
    }


# ── public API ───────────────────────────────────────────────────────────────


def embed_user_embeddings(settings, cases: list[str] | None = None) -> None:
    """Recompute and merge UserEmbedding rows for each case.

    For now this always recomputes from all available ClipEmbedding rows
    for the case. Eligibility filtering happens upstream at the clip
    embedding stage.

    TODO: guard with the universal idempotence/hash module once that
    module exists. ``settings`` is accepted for forward-compatibility
    even though no field is read today.
    """
    from modules.console import log
    from modules.database import Base, UserEmbedding, get_engine, get_session
    from modules.embeddings.cases import DEFAULT_CASES
    from modules.embeddings.state import (
        get_clip_embedding_rows_for_user_aggregation,
    )

    case_names = list(cases) if cases is not None else list(DEFAULT_CASES)
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for case in case_names:
            rows = get_clip_embedding_rows_for_user_aggregation(session, case)
            if not rows:
                log(f"embed:user:{case}", "nothing to do")
                continue

            aggregated = aggregate_user_embeddings_from_rows(rows)
            log(f"embed:user:{case}", f"{len(aggregated)} users to embed")

            for user_id, mean_blob in aggregated.items():
                row = UserEmbedding(
                    user_id=user_id,
                    embedding_case=case,
                    embedding=mean_blob,
                )
                session.merge(row)
                session.commit()

            log(f"embed:user:{case}", "done", level="ok")
    finally:
        session.close()
