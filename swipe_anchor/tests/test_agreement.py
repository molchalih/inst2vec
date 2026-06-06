"""Fleiss' kappa + nominal Krippendorff alpha headline numbers (design §3.1)."""

import math

from swipe_anchor.core.consensus import Vote, agreement


def test_perfect_agreement_is_one() -> None:
    items = {i: [Vote("a", 0), Vote("b", 0), Vote("c", 0)] for i in range(4)}
    out = agreement(items)
    assert math.isclose(out["fleiss_kappa"], 1.0, abs_tol=1e-9)
    assert math.isclose(out["krippendorff_alpha"], 1.0, abs_tol=1e-9)


def test_chance_level_agreement_near_zero() -> None:
    # Two raters maximally split across items → ~chance.
    items = {
        0: [Vote("a", 0), Vote("b", 1)],
        1: [Vote("a", 1), Vote("b", 0)],
        2: [Vote("a", 2), Vote("b", 0)],
        3: [Vote("a", 0), Vote("b", 2)],
    }
    out = agreement(items)
    assert out["fleiss_kappa"] < 0.2


def test_alpha_in_valid_range_and_perfect_is_one() -> None:
    perfect = {i: [Vote("a", 1), Vote("b", 1), Vote("c", 1)] for i in range(3)}
    assert math.isclose(agreement(perfect)["krippendorff_alpha"], 1.0, abs_tol=1e-9)
    # A noisy set: alpha must stay within [-1, 1].
    noisy = {0: [Vote("a", 0), Vote("b", 1)], 1: [Vote("a", 2), Vote("b", 0)]}
    a = agreement(noisy)["krippendorff_alpha"]
    assert -1.0 <= a <= 1.0
