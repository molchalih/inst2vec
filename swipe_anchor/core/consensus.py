"""Dawid-Skene at the odd-one-out level (design §3.1, refinement #3).

The latent is the 3-class odd-one-out label (which of the triple's sorted
creators is truly odd); ``skip`` is a first-class observed category, never a
missing value. D-S runs over these per-response labels — NOT over derived
triplets (one answer = two correlated triplets, so triplet-level D-S would
double-count and miscalibrate every reliability).

Pure + deterministic. The service maps creator ids <-> class indices 0/1/2.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

_CHANCE = 1.0 / 3.0


@dataclass(frozen=True)
class Vote:
    """One annotator's odd-one-out label: class 0/1/2, or ``None`` for skip."""

    annotator_id: str
    label: int | None


@dataclass(frozen=True)
class ItemConsensus:
    posterior: tuple[float, float, float]
    consensus_class: int | None
    posterior_max: float
    n_effective: float
    skip_rate: float
    agreement: float


def competence_from_reliability(reliability: float) -> float:
    """Map reliability in [0,1] to P(correct) in [1/3, 1] (chance is 1/3)."""
    r = min(1.0, max(0.0, reliability))
    return _CHANCE + r * (2.0 / 3.0)


def _posterior_from_competence(
    labels: Sequence[int], comps: Sequence[float]
) -> np.ndarray:
    """MAP posterior over the 3 classes from non-skip votes (uniform prior)."""
    loglik = np.zeros(3)
    for label, c in zip(labels, comps, strict=True):
        c = min(c, 1.0 - 1e-9)  # guard: c==1.0 → log(0) = -inf → NaN posterior
        wrong = (1.0 - c) / 2.0
        for t in range(3):
            loglik[t] += np.log(c if label == t else wrong)
    loglik -= loglik.max()
    post = np.exp(loglik)
    return post / post.sum()


def update_item(
    votes: Sequence[Vote],
    reliability: Mapping[str, float],
    *,
    skip_dominant: float = 0.5,
) -> ItemConsensus:
    """Single-item MAP given **fixed** reliabilities (the inline cheap update)."""
    n_total = len(votes)
    nonskip = [v for v in votes if v.label is not None]
    if not nonskip:
        return ItemConsensus((1 / 3, 1 / 3, 1 / 3), None, 1 / 3, 0.0, 1.0, 0.0)

    labels = [int(v.label) for v in nonskip]  # type: ignore[arg-type]
    comps = [competence_from_reliability(reliability.get(v.annotator_id, 0.5)) for v in nonskip]
    post = _posterior_from_competence(labels, comps)
    top = int(np.argmax(post))
    skip_rate = (n_total - len(nonskip)) / n_total
    n_eff = float(sum(reliability.get(v.annotator_id, 0.5) for v in nonskip))  # reliability-weighted vote count (plan §3 n_effective)
    agreement = labels.count(top) / len(labels)
    consensus_class = None if skip_rate >= skip_dominant else top
    return ItemConsensus(
        posterior=(float(post[0]), float(post[1]), float(post[2])),
        consensus_class=consensus_class,
        posterior_max=float(post[top]),
        n_effective=n_eff,
        skip_rate=skip_rate,
        agreement=agreement,
    )


@dataclass(frozen=True)
class EmResult:
    item_posteriors: dict[Hashable, tuple[float, float, float]]
    competence: dict[str, float]


def full_em(
    items: Mapping[Hashable, Sequence[Vote]],
    *,
    trusted: set[str],
    prior_reliability: float = 0.5,
    dirichlet_conc: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> EmResult:
    """Joint EM over all items: re-estimate per-annotator competence + per-item
    posterior. Stateless over the response history (recomputed each sweep).

    Untrusted annotators are pinned at the neutral-prior competence and excluded
    from the M-step (warm-up guard, refinement #2): sparse early data cannot
    produce degenerate reliabilities or let an uncalibrated voter steer others.
    """
    neutral = competence_from_reliability(prior_reliability)
    annotators = {v.annotator_id for votes in items.values() for v in votes}
    comp = {a: neutral for a in annotators}

    posteriors: dict[Hashable, np.ndarray] = {}
    for _ in range(max_iter):
        # E-step: posterior per item from current competences.
        for key, votes in items.items():
            nonskip = [v for v in votes if v.label is not None]
            if not nonskip:
                posteriors[key] = np.full(3, 1 / 3)
                continue
            labels = [int(v.label) for v in nonskip]  # type: ignore[arg-type]
            comps = [comp[v.annotator_id] for v in nonskip]
            posteriors[key] = _posterior_from_competence(labels, comps)

        # M-step: smoothed expected accuracy, trusted annotators only.
        correct = {a: 0.0 for a in annotators}
        total = {a: 0.0 for a in annotators}
        for key, votes in items.items():
            post = posteriors[key]
            for v in votes:
                if v.label is None:
                    continue
                correct[v.annotator_id] += float(post[int(v.label)])
                total[v.annotator_id] += 1.0

        max_delta = 0.0
        for a in annotators:
            if a not in trusted or total[a] == 0.0:
                continue
            new = (correct[a] + dirichlet_conc * _CHANCE) / (total[a] + dirichlet_conc)
            new = min(1.0, max(_CHANCE, new))
            max_delta = max(max_delta, abs(new - comp[a]))
            comp[a] = new
        if max_delta < tol:
            break

    return EmResult(
        item_posteriors={k: (float(p[0]), float(p[1]), float(p[2])) for k, p in posteriors.items()},
        competence=comp,
    )


# Four nominal categories for agreement: classes 0/1/2 plus skip (index 3).
_N_CATS = 4


def _category(label: int | None) -> int:
    return 3 if label is None else int(label)


def agreement(items: Mapping[Hashable, Sequence[Vote]]) -> dict[str, float]:
    """Fleiss' kappa + nominal Krippendorff's alpha over resolved items.

    ``skip`` is its own nominal category, so high-skip items lower agreement
    rather than being dropped. Items with < 2 votes are ignored (undefined).

    Fleiss' kappa assumes approximately equal raters per item; both metrics match
    reference implementations asymptotically (large total vote count) and may
    differ slightly from strict definitions on very small samples.
    """
    rows = []  # per-item category counts
    for votes in items.values():
        counts = [0] * _N_CATS
        for v in votes:
            counts[_category(v.label)] += 1
        if sum(counts) >= 2:
            rows.append(counts)
    if not rows:
        return {"fleiss_kappa": 0.0, "krippendorff_alpha": 0.0}

    # Fleiss' kappa
    n_items = len(rows)
    cat_totals = [0.0] * _N_CATS
    p_i_sum = 0.0
    grand = 0
    for counts in rows:
        n = sum(counts)
        grand += n
        agree = sum(c * (c - 1) for c in counts) / (n * (n - 1))
        p_i_sum += agree
        for k in range(_N_CATS):
            cat_totals[k] += counts[k]
    p_cat = [c / grand for c in cat_totals]
    p_bar = p_i_sum / n_items
    p_e = sum(p * p for p in p_cat)
    fleiss = 1.0 if p_e >= 1.0 else (p_bar - p_e) / (1.0 - p_e)

    # Nominal Krippendorff's alpha = 1 - Do/De, pooled estimator.
    do_num = sum(sum(c * (c - 1) for c in counts) for counts in rows)
    do_den = sum(sum(counts) * (sum(counts) - 1) for counts in rows)
    observed_disagree = 1.0 - (do_num / do_den)
    # Standard n/(n-1) correction (grand = total votes counted).
    expected_disagree = (1.0 - sum(p * p for p in p_cat)) * (grand / (grand - 1)) if grand > 1 else 0.0
    alpha = 1.0 if expected_disagree == 0 else 1.0 - observed_disagree / expected_disagree

    return {"fleiss_kappa": float(fleiss), "krippendorff_alpha": float(alpha)}
