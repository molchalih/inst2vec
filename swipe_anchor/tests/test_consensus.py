"""Odd-one-out-level Dawid-Skene (design §3.1). Classes are 0/1/2 = the sorted
position of the truly-odd creator; ``None`` label == skip (first-class category)."""

import math

from swipe_anchor.core.consensus import Vote, competence_from_reliability, update_item


def test_competence_maps_reliability_into_chance_to_one() -> None:
    assert competence_from_reliability(0.0) == 1 / 3
    assert competence_from_reliability(1.0) == 1.0
    assert math.isclose(competence_from_reliability(0.5), 1 / 3 + 0.5 * (2 / 3))


def test_unanimous_confident_class() -> None:
    votes = [Vote("a", 1), Vote("b", 1), Vote("c", 1)]
    rel = {"a": 0.9, "b": 0.9, "c": 0.9}
    est = update_item(votes, rel)
    assert est.consensus_class == 1
    assert est.posterior_max > 0.9
    assert est.skip_rate == 0.0
    assert est.agreement == 1.0


def test_skip_dominated_item_has_no_consensus() -> None:
    votes = [Vote("a", None), Vote("b", None), Vote("c", 2)]
    rel = {"a": 0.9, "b": 0.9, "c": 0.9}
    est = update_item(votes, rel)
    assert est.consensus_class is None  # >= 50% skip → near-tie / ambiguous
    assert math.isclose(est.skip_rate, 2 / 3)


def test_no_votes_is_uninformative() -> None:
    est = update_item([], {})
    assert est.consensus_class is None
    assert est.n_effective == 0.0


def test_split_vote_favors_higher_reliability_side() -> None:
    # Two voters cross class 0, one (less reliable) crosses class 2.
    votes = [Vote("a", 0), Vote("b", 0), Vote("c", 2)]
    rel = {"a": 0.9, "b": 0.9, "c": 0.5}
    est = update_item(votes, rel)
    assert est.consensus_class == 0
    assert 0.0 < est.posterior_max < 1.0
    assert est.agreement == 2 / 3  # two of three nonskip voters on the top class


def test_competence_one_does_not_nan_on_disagreement() -> None:
    votes = [Vote("a", 0), Vote("b", 1)]
    est = update_item(votes, {"a": 1.0, "b": 1.0})
    # Must not be NaN; with equal max-competence disagreement it stays finite.
    assert est.consensus_class in (0, 1)
    assert est.posterior_max == est.posterior_max  # not NaN
