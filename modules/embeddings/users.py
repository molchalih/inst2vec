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
