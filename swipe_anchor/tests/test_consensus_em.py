"""Full EM recovers planted labels + separates a griefer (design §3.1)."""

import numpy as np

from swipe_anchor.core.consensus import Vote, full_em


def _planted_crowd(seed: int = 0):
    """6 items, 5 good annotators (~0.9 correct) + 1 griefer (random)."""
    rng = np.random.default_rng(seed)
    truth = [0, 1, 2, 0, 1, 2]
    good = [f"g{i}" for i in range(5)]
    items: dict[int, list[Vote]] = {}
    for item, t in enumerate(truth):
        votes = []
        for a in good:
            label = t if rng.random() < 0.9 else int(rng.integers(3))
            votes.append(Vote(a, label))
        votes.append(Vote("griefer", int(rng.integers(3))))
        items[item] = votes
    return truth, items, good


def test_em_recovers_truth_and_demotes_griefer() -> None:
    truth, items, good = _planted_crowd()
    res = full_em(items, trusted=set(good) | {"griefer"}, prior_reliability=0.5)
    recovered = [int(np.argmax(res.item_posteriors[i])) for i in range(len(truth))]
    assert recovered == truth
    mean_good = np.mean([res.competence[a] for a in good])
    assert res.competence["griefer"] < mean_good


def test_untrusted_annotator_competence_is_pinned() -> None:
    _, items, good = _planted_crowd()
    res = full_em(items, trusted=set(good), prior_reliability=0.5)
    # griefer is NOT in trusted → its competence stays at the neutral prior.
    from swipe_anchor.core.consensus import competence_from_reliability

    assert res.competence["griefer"] == competence_from_reliability(0.5)
