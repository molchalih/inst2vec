"""Tunable knobs for the balancer + consensus backbone (design §5)."""

import pytest

from swipe_anchor.config import Settings


def test_defaults_match_design() -> None:
    s = Settings()
    assert s.min_overlap == 5
    assert s.max_overlap == 9
    assert s.confidence_threshold == 0.80
    assert s.warmup_k == 8
    assert s.p_gold == 0.125
    assert s.max_inflight == 2
    assert s.w_info == 0.0  # Phase-4 seam: off now


def test_from_env_overrides_and_coerces() -> None:
    s = Settings.from_env({"SA_MAX_INFLIGHT": "5", "SA_CONFIDENCE_THRESHOLD": "0.9"})
    assert s.max_inflight == 5
    assert s.confidence_threshold == 0.9
    # untouched knobs keep defaults
    assert s.min_overlap == 5


def test_from_env_bad_value_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="SA_MAX_INFLIGHT"):
        Settings.from_env({"SA_MAX_INFLIGHT": "not-an-int"})
