"""gather_stats aggregates totals, a time series, and ranked contributors."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from swipe_anchor.backend.stats import gather_stats
from swipe_anchor.db import create_app_engine, make_session_factory, session_scope
from swipe_anchor.db.models import (
    AccessCode,
    Annotator,
    Assignment,
    Comparison,
    Consensus,
    Response,
    Triplet,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _factory():
    engine = create_app_engine("sqlite:///:memory:")
    factory = make_session_factory(engine)

    @contextmanager
    def sf():
        with session_scope(factory) as s:
            yield s

    return sf


def _seed(s) -> None:
    s.add(AccessCode(code="tg_aaa", note="dasha", is_active=True))
    s.add(Annotator(annotator_id="tg_aaa", reliability=0.9, n_gold_seen=4, n_gold_correct=3))
    s.add(Annotator(annotator_id="bob", reliability=0.5))
    for i, status in enumerate(["retired", "retired", "ambiguous", "open", "gold"]):
        s.add(
            Comparison(
                comparison_id=f"c{i}", creator_a=1, creator_b=2, creator_c=3, status=status
            )
        )
    s.add(Consensus(comparison_id="c0", consensus_odd=3, agreement=1.0, resolved=True))
    s.add(Consensus(comparison_id="c1", consensus_odd=2, agreement=0.6, resolved=True))
    s.add(Triplet(anchor_id=1, positive_id=2, negative_id=3, comparison_id="c0"))
    s.add(Triplet(anchor_id=2, positive_id=1, negative_id=3, comparison_id="c1"))
    # 3 judgments from dasha (c0/c1/c2), 1 from bob (c3) — distinct (comparison,
    # annotator) pairs satisfy the assignment uniqueness + FK constraints, and the
    # increasing timestamps give a deterministic cumulative series.
    rows = [("tg_aaa", "c0", 0), ("tg_aaa", "c1", 1), ("tg_aaa", "c2", 2), ("bob", "c3", 3)]
    for i, (ann, cmp_id, mins) in enumerate(rows):
        s.add(
            Assignment(assignment_id=f"asg{i}", comparison_id=cmp_id, annotator_id=ann)
        )
        s.add(
            Response(
                response_id=f"r{i}",
                assignment_id=f"asg{i}",
                comparison_id=cmp_id,
                annotator_id=ann,
                created_at=T0 + timedelta(minutes=mins),
            )
        )


def test_gather_stats_totals() -> None:
    sf = _factory()
    with sf() as s:
        _seed(s)
    with sf() as s:
        st = gather_stats(s)
    t = st["totals"]
    assert t["responses"] == 4
    assert t["triplets"] == 2
    assert t["annotators"] == 2
    assert t["comparisons"] == {"open": 1, "retired": 2, "ambiguous": 1, "gold": 1}
    assert t["resolved"] == 3
    assert abs(t["mean_agreement"] - 0.8) < 1e-9  # (1.0 + 0.6) / 2
    assert 0.0 <= t["mean_reliability"] <= 1.0
    assert t["gold_seen"] == 4 and t["gold_correct"] == 3


def test_gather_stats_timeseries_sorted() -> None:
    sf = _factory()
    with sf() as s:
        _seed(s)
    with sf() as s:
        st = gather_stats(s)
    times = st["response_times"]
    assert len(times) == 4
    assert times == sorted(times)  # ascending


def test_gather_stats_contributors_ranked_and_labelled() -> None:
    sf = _factory()
    with sf() as s:
        _seed(s)
    with sf() as s:
        st = gather_stats(s)
    contrib = st["per_annotator"]
    assert contrib[0] == {"label": "dasha", "n": 3}  # operator note used, top of list
    assert {c["label"] for c in contrib} == {"dasha", "bob"}


def test_gather_stats_empty() -> None:
    sf = _factory()
    with sf() as s:
        st = gather_stats(s)
    assert st["totals"]["responses"] == 0
    assert st["response_times"] == []
    assert st["per_annotator"] == []
    assert st["totals"]["mean_agreement"] == 0.0
