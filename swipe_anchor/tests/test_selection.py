"""Bias-guard representative-clip selection (plan §6.1).

The whole anchor is only independent if what the human sees is picked by a
modality-neutral rule: medoid in the STANDARDIZED late-fusion space (so the
visual block can't shout), plus farthest-point spanning — never the biased
``user_clusters.centrality``. These tests pin that contract.
"""

import numpy as np

from swipe_anchor.core.selection import select_representative_clips, standardize


def test_medoid_is_clip_nearest_the_centroid() -> None:
    vectors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [3.0, 3.0]])
    clip_ids = [100, 101, 102, 103]

    picks = select_representative_clips(vectors, clip_ids, n=1)

    assert picks == [103]  # [3,3] is nearest the centroid [3.25, 3.25]


def test_spanning_uses_farthest_point_after_medoid() -> None:
    vectors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [3.0, 3.0]])
    clip_ids = [100, 101, 102, 103]

    picks = select_representative_clips(vectors, clip_ids, n=3)

    # medoid first, then greedy max-min distance spanning the creator's range.
    assert picks[0] == 103
    assert set(picks) == {103, 101, 102}
    assert len(picks) == 3


def test_selection_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(12, 8))
    clip_ids = list(range(200, 212))

    a = select_representative_clips(vectors, clip_ids, n=3)
    b = select_representative_clips(vectors, clip_ids, n=3)

    assert a == b


def test_n_larger_than_available_returns_all() -> None:
    vectors = np.array([[0.0], [1.0]])
    picks = select_representative_clips(vectors, [1, 2], n=5)
    assert set(picks) == {1, 2}


def test_standardize_guards_zero_variance_dimension() -> None:
    vectors = np.array([[5.0, 7.0]])
    mean = np.array([0.0, 0.0])
    std = np.array([1.0, 0.0])  # second dim has zero variance dataset-wide

    out = standardize(vectors, mean, std)

    assert np.isfinite(out).all()
    assert out[0, 0] == 5.0
    assert out[0, 1] == 7.0  # divided by guarded std of 1, not 0
