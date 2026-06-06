"""Gold-anchored reliability blend + warm-up guard + behavioral penalty (§3.2)."""

import math

from swipe_anchor.core.reliability import (
    behavioral_factor,
    gold_accuracy,
    gold_blend_lambda,
    is_trusted,
    reliability,
)


def test_gold_accuracy_uses_beta_prior() -> None:
    # No data → prior mean alpha0/(alpha0+beta0) = 0.5 for (2,2).
    assert math.isclose(gold_accuracy(0, 0, 2.0, 2.0), 0.5)
    # 8/10 correct, prior (2,2) → (2+8)/(2+2+10) = 10/14.
    assert math.isclose(gold_accuracy(10, 8, 2.0, 2.0), 10 / 14)


def test_lambda_starts_at_gold_and_decays_to_floor() -> None:
    assert gold_blend_lambda(0, warmup_k=8, floor=0.4) == 1.0
    assert gold_blend_lambda(1000, warmup_k=8, floor=0.4) == 0.4


def test_warmup_guard() -> None:
    assert not is_trusted(n_eff=3, warmup_k=8)
    assert is_trusted(n_eff=8, warmup_k=8)


def test_behavioral_factor_penalizes_too_fast_and_no_dwell() -> None:
    clean = behavioral_factor(min_rt_ms=2000, mean_dwell_ms=1500, ever_expanded=True, constant_streak=1)
    bad = behavioral_factor(min_rt_ms=120, mean_dwell_ms=0, ever_expanded=False, constant_streak=50)
    assert clean == 1.0
    assert bad < 0.5


def test_reliability_blend_is_gold_dominated_when_cold() -> None:
    # Cold (n_eff=0) → pure gold (lambda=1): ds_comp is ignored.
    r = reliability(gold_acc=0.9, ds_comp=0.1, n_eff=0, warmup_k=8, behavioral=1.0, floor=0.4)
    assert math.isclose(r, 0.9)
