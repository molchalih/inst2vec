"""Tests for the odd-one-out -> triplet derivation rule (plan §1.2, §1.3)."""

import pytest

from swipe_anchor.core.triplets import Triplet, derive_triplets


def test_one_answer_yields_two_triplets() -> None:
    # Crossing C out of (A, B, C) asserts A and B are mutually nearer than C.
    triplets = derive_triplets((10, 20, 30), odd_id=30)

    assert triplets == [
        Triplet(anchor=10, positive=20, negative=30),
        Triplet(anchor=20, positive=10, negative=30),
    ]


def test_odd_can_be_any_of_the_three() -> None:
    triplets = derive_triplets((10, 20, 30), odd_id=10)

    assert triplets == [
        Triplet(anchor=20, positive=30, negative=10),
        Triplet(anchor=30, positive=20, negative=10),
    ]


def test_odd_not_in_triple_is_rejected() -> None:
    with pytest.raises(ValueError, match="odd_id"):
        derive_triplets((10, 20, 30), odd_id=99)


def test_triple_must_have_three_distinct_creators() -> None:
    with pytest.raises(ValueError, match="distinct"):
        derive_triplets((10, 10, 30), odd_id=30)
