"""Per-item consensus + retirement + triplet materialization (design §3.8, §4)."""

import uuid

import pytest
from sqlalchemy.orm import Session

from swipe_anchor.backend.consensus_writer import recompute_item
from swipe_anchor.config import Settings
from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import (
    Annotator,
    Assignment,
    Comparison,
    Consensus,
    Response,
    Triplet,
)


@pytest.fixture
def session() -> Session:
    engine = create_app_engine("sqlite:///:memory:")
    with Session(engine) as s:
        yield s


def _vote(s: Session, cmp_id: str, ann: str, odd: int | None, rel: float = 0.9) -> None:
    if s.get(Annotator, ann) is None:
        s.add(Annotator(annotator_id=ann, reliability=rel))
    assignment_id = str(uuid.uuid4())
    s.add(
        Assignment(
            assignment_id=assignment_id,
            comparison_id=cmp_id,
            annotator_id=ann,
        )
    )
    s.flush()  # ensure Assignment exists before Response FK
    s.add(
        Response(
            response_id=str(uuid.uuid4()),
            assignment_id=assignment_id,
            comparison_id=cmp_id,
            annotator_id=ann,
            odd_creator_id=odd,
        )
    )
    s.flush()


def test_confident_consensus_retires_and_materializes_two_triplets(session: Session) -> None:
    session.add(Comparison(comparison_id="c1", creator_a=10, creator_b=20, creator_c=30, target_k=3))
    session.flush()
    for ann in ("a", "b", "c"):
        _vote(session, "c1", ann, odd=30)  # everyone crosses creator 30
    recompute_item(session, "c1", Settings(min_overlap=3))

    cmp = session.get(Comparison, "c1")
    assert cmp.status == "retired"
    cons = session.get(Consensus, "c1")
    assert cons.consensus_odd == 30 and cons.resolved is True
    trips = session.query(Triplet).filter_by(comparison_id="c1").all()
    assert len(trips) == 2
    assert {(t.anchor_id, t.positive_id, t.negative_id) for t in trips} == {
        (10, 20, 30),
        (20, 10, 30),
    }


def test_ambiguous_item_rides_to_cap_then_flags_no_triplet(session: Session) -> None:
    session.add(Comparison(comparison_id="c2", creator_a=1, creator_b=2, creator_c=3, target_k=2))
    session.flush()
    # Skip-dominated → never confident.
    _vote(session, "c2", "a", odd=None)
    _vote(session, "c2", "b", odd=None)
    _vote(session, "c2", "c", odd=1)
    recompute_item(session, "c2", Settings(min_overlap=2, max_overlap=3))
    cmp = session.get(Comparison, "c2")
    assert cmp.status == "ambiguous"
    assert session.query(Triplet).filter_by(comparison_id="c2").count() == 0
    cons = session.get(Consensus, "c2")
    assert cons is not None and cons.resolved is False and cons.consensus_odd is None


def test_below_quorum_stays_open(session: Session) -> None:
    session.add(Comparison(comparison_id="c3", creator_a=1, creator_b=2, creator_c=3, target_k=5))
    session.flush()
    _vote(session, "c3", "a", odd=3)
    recompute_item(session, "c3", Settings(min_overlap=5, max_overlap=9))
    assert session.get(Comparison, "c3").status == "open"


def test_retired_triplets_refresh_when_reliability_drifts(session: Session) -> None:
    session.add(Comparison(comparison_id="c4", creator_a=10, creator_b=20, creator_c=30, target_k=2))
    session.flush()
    settings = Settings(min_overlap=2, confidence_threshold=0.6, max_overlap=10)
    _vote(session, "c4", "a", odd=30, rel=0.9)
    _vote(session, "c4", "b", odd=30, rel=0.9)
    recompute_item(session, "c4", settings)
    old = session.query(Triplet).filter_by(comparison_id="c4").all()
    assert session.get(Comparison, "c4").status == "retired"
    assert len(old) > 0 and old[0].weight >= 0.6

    # Demote the two voters to near-chance: the item is now under-confident.
    session.get(Annotator, "a").reliability = 0.01
    session.get(Annotator, "b").reliability = 0.01
    session.flush()
    recompute_item(session, "c4", settings)

    trips = session.query(Triplet).filter_by(comparison_id="c4").all()
    cons = session.get(Consensus, "c4")
    # Triplets were refreshed to the CURRENT (lower) confidence, not left stale.
    assert all(t.weight < 0.6 for t in trips)
    # ...and stay consistent with the current consensus_odd.
    if cons.consensus_odd is not None:
        assert {t.negative_id for t in trips} == {cons.consensus_odd}
    else:
        assert len(trips) == 0
