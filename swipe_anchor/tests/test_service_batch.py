"""Scored next_batch: max_inflight, expiry-aware inflight, gold injection (§4)."""

import random
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from swipe_anchor.backend.service import next_batch
from swipe_anchor.config import Settings
from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import Assignment, Comparison, GoldItem


@pytest.fixture
def session() -> Session:
    engine = create_app_engine("sqlite:///:memory:")
    with Session(engine) as s:
        yield s


def _cmp(s: Session, cid: str, status: str = "open", kind: str = "random") -> None:
    s.add(Comparison(comparison_id=cid, creator_a=1, creator_b=2, creator_c=3, kind=kind, status=status, target_k=5))


def test_max_inflight_blocks_overlap(session: Session) -> None:
    _cmp(session, "c1")
    session.flush()
    s = Settings(max_inflight=1)
    now = datetime.now(UTC)
    next_batch(session, "ann-1", n=1, rng=random.Random(0), settings=s, now=now)
    picks = next_batch(session, "ann-2", n=1, rng=random.Random(0), settings=s, now=now)
    assert picks == []


def test_expired_inflight_does_not_block(session: Session) -> None:
    _cmp(session, "c1")
    session.flush()
    s = Settings(max_inflight=1)
    past = datetime.now(UTC) - timedelta(hours=1)
    next_batch(session, "ann-1", n=1, rng=random.Random(0), settings=s, now=past)
    now = datetime.now(UTC)
    picks = next_batch(session, "ann-2", n=1, rng=random.Random(0), settings=s, now=now)
    assert {c.comparison_id for c in picks} == {"c1"}


def test_assignment_carries_expiry(session: Session) -> None:
    _cmp(session, "c1")
    session.flush()
    now = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    next_batch(session, "ann-1", n=1, rng=random.Random(0), settings=Settings(assign_ttl_s=600), now=now)
    asg = session.query(Assignment).filter_by(annotator_id="ann-1").one()
    assert asg.expires_at == now + timedelta(seconds=600)


def test_gold_injected_when_forced(session: Session) -> None:
    _cmp(session, "g1", status="gold")
    session.add(GoldItem(comparison_id="g1", known_odd=3))
    session.flush()
    picks = next_batch(session, "ann-1", n=1, rng=random.Random(0), settings=Settings(p_gold=1.0), now=datetime.now(UTC))
    assert {c.comparison_id for c in picks} == {"g1"}


def test_gold_exposure_cap_limits_distinct_gold(session: Session) -> None:
    for gid in ("g1", "g2", "g3"):
        session.add(Comparison(comparison_id=gid, creator_a=1, creator_b=2, creator_c=3, status="gold", target_k=5))
        session.add(GoldItem(comparison_id=gid, known_odd=3))
    session.flush()
    s = Settings(p_gold=1.0, gold_exposure_cap=2)
    now = datetime.now(UTC)
    seen_gold: set[str] = set()
    for _ in range(5):
        picks = next_batch(session, "ann", n=1, rng=random.Random(0), settings=s, now=now)
        session.flush()
        for c in picks:
            if session.get(GoldItem, c.comparison_id) is not None:
                seen_gold.add(c.comparison_id)
    assert len(seen_gold) == 2  # capped at 2 DISTINCT gold items for this annotator


def test_sweep_reclaim_frees_item_for_another_annotator(session: Session) -> None:
    from swipe_anchor.backend.sweep import run_sweep

    _cmp(session, "c1")
    session.flush()
    s = Settings(max_inflight=1, assign_ttl_s=600)
    past = datetime.now(UTC) - timedelta(hours=1)
    next_batch(session, "ann-1", n=1, rng=random.Random(0), settings=s, now=past)
    session.flush()
    now = datetime.now(UTC)
    run_sweep(session, s, now=now)  # reclaims ann-1's expired hold → status "expired"
    session.flush()
    picks = next_batch(session, "ann-2", n=1, rng=random.Random(0), settings=s, now=now)
    assert {c.comparison_id for c in picks} == {"c1"}


def test_unresolved_item_past_target_k_stays_eligible(session: Session) -> None:
    # [P1] target_k column = 5 but settings.max_overlap = 9. A split item that hit
    # its target_k votes without confidence stays "open" and MUST remain draftable
    # up to max_overlap (so it can still reach confidence or be flagged ambiguous).
    c = Comparison(comparison_id="c1", creator_a=1, creator_b=2, creator_c=3, target_k=5, status="open")
    c.n_judgments = 5
    session.add(c)
    session.flush()
    picks = next_batch(session, "ann-new", n=1, rng=random.Random(0), settings=Settings(max_overlap=9), now=datetime.now(UTC))
    assert {p.comparison_id for p in picks} == {"c1"}


def test_item_at_max_overlap_drops_out_of_pool(session: Session) -> None:
    # [P1] At/over the hard cap it must leave the draw (recompute_item flags it
    # ambiguous there); the balancer must not keep over-sampling it.
    c = Comparison(comparison_id="c1", creator_a=1, creator_b=2, creator_c=3, target_k=5, status="open")
    c.n_judgments = 9
    session.add(c)
    session.flush()
    picks = next_batch(session, "ann-new", n=1, rng=random.Random(0), settings=Settings(max_overlap=9), now=datetime.now(UTC))
    assert picks == []
