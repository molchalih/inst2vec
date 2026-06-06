"""Modality-neutral representative-clip selection — the bias guard (plan §6.1).

Selecting *which clips a human sees* with the visually-biased embedding would let
the human re-inherit the bias and the anchor would stop being independent
(plan §12 "bias leakage"). So:

1. Work in the **standardized** late-fusion space (z-scored per dimension across
   the whole dataset) — the ``embeddings_v2.md`` §3 fix that stops the visual
   block from shouting. ``standardize`` takes dataset-wide ``mean``/``std``.
2. Pick the **medoid** (clip nearest the creator centroid in that space), then
   add spanning clips by greedy farthest-point sampling — a deterministic,
   modality-neutral rule. Never rank by ``user_clusters.centrality``.

Everything here is pure numpy and fully deterministic for a fixed input order.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def standardize(vectors: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score ``vectors`` with dataset-wide ``mean``/``std``.

    Zero-variance dimensions are guarded (treated as std 1) so the output stays
    finite — a constant dimension simply contributes its centered value.
    """
    safe_std = np.where(std == 0, 1.0, std)
    return (vectors - mean) / safe_std


def select_representative_clips(
    std_vectors: np.ndarray,
    clip_ids: Sequence[int],
    n: int = 3,
) -> list[int]:
    """Return up to ``n`` representative clip ids: medoid first, then spanning.

    ``std_vectors[i]`` is the (already standardized) vector for ``clip_ids[i]``.
    Selection is deterministic: ties resolve to the lowest index.
    """
    if len(clip_ids) != len(std_vectors):
        raise ValueError("std_vectors and clip_ids must align")
    count = len(clip_ids)
    if count == 0:
        return []
    n = min(n, count)

    centroid = std_vectors.mean(axis=0)
    dist_to_centroid = np.linalg.norm(std_vectors - centroid, axis=1)
    medoid_idx = int(np.argmin(dist_to_centroid))

    chosen = [medoid_idx]
    while len(chosen) < n:
        chosen_pts = std_vectors[chosen]
        # distance of every point to the nearest already-chosen point
        d = np.linalg.norm(
            std_vectors[:, None, :] - chosen_pts[None, :, :], axis=2
        ).min(axis=1)
        d[chosen] = -np.inf  # never re-pick
        chosen.append(int(np.argmax(d)))

    return [clip_ids[i] for i in chosen]
