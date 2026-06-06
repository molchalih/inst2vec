"""Schema roundtrip + key-constraint tests for the app store (plan §3)."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import (
    Annotator,
    Assignment,
    Comparison,
    Creator,
    Response,
)


@pytest.fixture
def session() -> Session:
    engine = create_app_engine("sqlite:///:memory:")
    with Session(engine) as s:
        yield s


def test_creator_comparison_roundtrip(session: Session) -> None:
    session.add(Creator(creator_id=1, seed_group="Artist", digest_version=1))
    session.add(
        Comparison(
            comparison_id="cmp-1",
            creator_a=1,
            creator_b=2,
            creator_c=3,
            kind="random",
            seed_group="Artist",
            target_k=5,
        )
    )
    session.commit()

    cmp = session.get(Comparison, "cmp-1")
    assert cmp is not None
    assert (cmp.creator_a, cmp.creator_b, cmp.creator_c) == (1, 2, 3)
    # Sensible denormalized defaults so the balancer can read them immediately.
    assert cmp.n_judgments == 0
    assert cmp.status == "open"


def test_annotator_reliability_defaults(session: Session) -> None:
    session.add(Annotator(annotator_id="ann-1"))
    session.commit()

    ann = session.get(Annotator, "ann-1")
    assert ann is not None
    assert ann.reliability == pytest.approx(0.5)
    assert ann.n_responses == 0
    assert ann.is_blocked is False


def test_assignment_unique_per_annotator_and_comparison(session: Session) -> None:
    session.add(
        Comparison(comparison_id="cmp-1", creator_a=1, creator_b=2, creator_c=3)
    )
    session.add(Annotator(annotator_id="ann-1"))
    session.add(
        Assignment(assignment_id="asg-1", comparison_id="cmp-1", annotator_id="ann-1")
    )
    session.commit()

    # The same annotator must not be assigned the same comparison twice.
    session.add(
        Assignment(assignment_id="asg-2", comparison_id="cmp-1", annotator_id="ann-1")
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_response_skip_is_first_class_null(session: Session) -> None:
    session.add(
        Comparison(comparison_id="cmp-1", creator_a=1, creator_b=2, creator_c=3)
    )
    session.add(Annotator(annotator_id="ann-1"))
    session.add(
        Assignment(assignment_id="asg-1", comparison_id="cmp-1", annotator_id="ann-1")
    )
    session.add(
        Response(
            response_id="rsp-1",
            assignment_id="asg-1",
            comparison_id="cmp-1",
            annotator_id="ann-1",
            odd_creator_id=None,  # skip / too close to call
        )
    )
    session.commit()

    rsp = session.scalar(select(Response).where(Response.response_id == "rsp-1"))
    assert rsp is not None
    assert rsp.odd_creator_id is None


def test_expires_at_roundtrips_as_utc_aware(session: Session) -> None:
    """SQLite drops tzinfo; UTCDateTime must re-attach UTC so comparisons work."""
    import uuid
    from datetime import UTC, datetime

    session.add(Comparison(comparison_id="c1", creator_a=1, creator_b=2, creator_c=3))
    session.add(Annotator(annotator_id="ann"))
    when = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)
    session.add(
        Assignment(
            assignment_id=str(uuid.uuid4()),
            comparison_id="c1",
            annotator_id="ann",
            status="issued",
            expires_at=when,
        )
    )
    session.flush()
    session.expire_all()
    got = session.query(Assignment).one().expires_at
    assert got.tzinfo is not None
    assert got == when  # tz-aware equality holds after readback
