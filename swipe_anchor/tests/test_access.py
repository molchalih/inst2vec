"""Access-code admission + assignment ownership (auth feature)."""

import random

import pytest
from sqlalchemy.orm import Session

from swipe_anchor.backend.service import (
    ForbiddenError,
    UnknownAssignmentError,
    is_access_allowed,
    next_batch,
    record_response,
)
from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import AccessCode, Assignment, Comparison


@pytest.fixture
def session() -> Session:
    engine = create_app_engine("sqlite:///:memory:")
    with Session(engine) as s:
        yield s


def test_empty_table_admits_any_nonempty_code(session: Session) -> None:
    assert is_access_allowed(session, "ANYTHING") is True
    assert is_access_allowed(session, "") is False
    assert is_access_allowed(session, "   ") is False


def test_with_rows_only_listed_active_codes_admitted(session: Session) -> None:
    session.add(AccessCode(code="GOOD", note="my friend dasha", is_active=True))
    session.add(AccessCode(code="OFF", note="revoked", is_active=False))
    session.commit()

    assert is_access_allowed(session, "GOOD") is True
    assert is_access_allowed(session, "OFF") is False
    assert is_access_allowed(session, "UNKNOWN") is False


def test_respond_rejects_a_foreign_assignment(session: Session) -> None:
    session.add(
        Comparison(comparison_id="c1", creator_a=10, creator_b=20, creator_c=30)
    )
    session.commit()
    next_batch(session, "owner-code", n=1, rng=random.Random(0))
    session.commit()
    asg = session.query(Assignment).filter_by(comparison_id="c1").one()

    with pytest.raises(ForbiddenError):
        record_response(
            session, asg.assignment_id, odd_id=30, expected_annotator_id="someone-else"
        )

    # The rightful owner can still answer it.
    result = record_response(
        session, asg.assignment_id, odd_id=30, expected_annotator_id="owner-code"
    )
    assert result.accepted is True


def test_respond_unknown_assignment_still_raises(session: Session) -> None:
    with pytest.raises(UnknownAssignmentError):
        record_response(session, "nope", odd_id=1, expected_annotator_id="x")
