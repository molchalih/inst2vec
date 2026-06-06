"""Pure scored draw (design §3.3). w_info is 0 now (Phase-4 seam)."""

import random

from swipe_anchor.config import Settings
from swipe_anchor.core.balancer import (
    Features,
    score,
    softmax_sample_without_replacement,
)


def test_score_weights_each_term() -> None:
    s = Settings()
    high = Features(coverage_gap=1.0, is_boundary=1.0, novelty=1.0, information=1.0)
    low = Features(coverage_gap=0.0, is_boundary=0.0, novelty=0.0, information=1.0)
    # information has weight 0 now, so the all-zero-but-info item scores ~eps only.
    assert score(high, s) > score(low, s)
    assert abs(score(low, s) - s.eps_random) < 1e-9


def test_softmax_sample_is_deterministic_and_sized() -> None:
    items = [(f"c{i}", float(i)) for i in range(10)]
    picks = softmax_sample_without_replacement(items, 3, temperature=0.5, rng=random.Random(0))
    again = softmax_sample_without_replacement(items, 3, temperature=0.5, rng=random.Random(0))
    assert picks == again
    assert len(picks) == 3
    assert len(set(picks)) == 3  # no replacement


def test_softmax_sample_caps_at_pool_size() -> None:
    items = [("a", 1.0), ("b", 2.0)]
    picks = softmax_sample_without_replacement(items, 5, temperature=0.5, rng=random.Random(0))
    assert set(picks) == {"a", "b"}
