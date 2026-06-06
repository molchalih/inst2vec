"""Gold-anchored, warm-up-guarded reliability (design §3.2, refinements #1, #2).

Dawid-Skene competence is weakly identified early and a collectively-biased
crowd can look reliable, so reliability blends an ABSOLUTE reference (gold
catch-trials) with the RELATIVE D-S competence, gold-weighted when cold. An
annotator is not trusted (nor allowed to steer others' estimates) until they
clear ``warmup_k`` gold+overlap judgments. Behavioral sanity multiplies the
result down but is never the sole signal. Pure functions.
"""

from __future__ import annotations


def gold_accuracy(n_gold_seen: int, n_gold_correct: int, alpha0: float, beta0: float) -> float:
    """Posterior-mean gold accuracy under a Beta(alpha0, beta0) prior."""
    return (alpha0 + n_gold_correct) / (alpha0 + beta0 + n_gold_seen)


def gold_blend_lambda(n_eff: float, warmup_k: int, floor: float = 0.4) -> float:
    """Weight on gold vs D-S: 1.0 when cold, decaying to ``floor`` as data grows."""
    if n_eff <= 0:
        return 1.0
    decayed = 1.0 - (1.0 - floor) * min(1.0, n_eff / (2.0 * warmup_k))
    return max(floor, decayed)


def is_trusted(n_eff: float, warmup_k: int) -> bool:
    return n_eff >= warmup_k


def behavioral_factor(
    *, min_rt_ms: int | None, mean_dwell_ms: float, ever_expanded: bool, constant_streak: int
) -> float:
    """Multiplicative sanity factor in (0, 1]; 1.0 means no concern."""
    factor = 1.0
    if min_rt_ms is not None and min_rt_ms < 300:  # implausibly fast
        factor *= 0.4
    if mean_dwell_ms <= 0:  # never actually looked at a card
        factor *= 0.5
    if not ever_expanded:  # mild prior; expanding is optional
        factor *= 0.9
    if constant_streak >= 20:  # same answer over and over
        factor *= 0.5
    return factor


def reliability(
    *,
    gold_acc: float,
    ds_comp: float,
    n_eff: float,
    warmup_k: int,
    behavioral: float,
    floor: float = 0.4,
) -> float:
    """Blend gold (absolute) + D-S competence (relative), then apply sanity."""
    lam = gold_blend_lambda(n_eff, warmup_k, floor)
    base = lam * gold_acc + (1.0 - lam) * ds_comp
    return min(1.0, max(0.0, base * behavioral))
