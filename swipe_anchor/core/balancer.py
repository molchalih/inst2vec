"""Scored, softmax-sampled draw over eligible comparisons (design §3.3, plan §4.1).

Pure: the service precomputes ``Features`` from the DB and supplies the RNG. The
``information`` term is wired but weight-0 this phase — Phase 4 turns it on by
populating ``comparisons.information`` and raising ``Settings.w_info``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from swipe_anchor.config import Settings


@dataclass(frozen=True)
class Features:
    coverage_gap: float  # (target_k - n_judgments)/target_k, in [0,1]
    is_boundary: float   # 1.0 if kind == "boundary" else 0.0
    novelty: float       # unseen-seed-group bonus for this annotator, in [0,1]
    information: float    # normalized active-selection score, in [0,1] (0 until Phase 4)


def score(f: Features, s: Settings) -> float:
    return (
        s.w_cover * f.coverage_gap
        + s.w_bound * f.is_boundary
        + s.w_novel * f.novelty
        + s.w_info * f.information
        + s.eps_random
    )


def softmax_sample_without_replacement[T](
    items: Sequence[tuple[T, float]],
    m: int,
    *,
    temperature: float,
    rng: random.Random,
) -> list[T]:
    """Draw up to ``m`` items by softmax weight, without replacement.

    Sampling (not argmax) so N concurrent annotators get a diverse high-score
    batch rather than all colliding on the single best item (plan §4.2).
    """
    pool = list(items)
    picks: list[T] = []
    for _ in range(min(m, len(pool))):
        weights = [math.exp(sc / temperature) for _, sc in pool]
        total = sum(weights)
        r = rng.random() * total
        cum = 0.0
        idx = len(pool) - 1
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                idx = i
                break
        picks.append(pool.pop(idx)[0])
    return picks
