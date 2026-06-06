"""schedule_reminder upserts one pending row; pop_due_reminders drains due ones."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from swipe_anchor.backend.reminders import pop_due_reminders, schedule_reminder
from swipe_anchor.db import create_app_engine, make_session_factory, session_scope
from swipe_anchor.db.models import Reminder

NOW = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)


def _factory():
    engine = create_app_engine("sqlite:///:memory:")
    factory = make_session_factory(engine)

    @contextmanager
    def sf():
        with session_scope(factory) as s:
            yield s

    return sf


def _count(s) -> int:
    return int(s.scalar(select(func.count()).select_from(Reminder)))


def test_schedule_sets_due_time() -> None:
    sf = _factory()
    with sf() as s:
        due = schedule_reminder(s, 42, hours=20, now=NOW)
        assert due == NOW + timedelta(hours=20)
    with sf() as s:
        assert _count(s) == 1


def test_reschedule_updates_the_single_pending_row() -> None:
    sf = _factory()
    with sf() as s:
        schedule_reminder(s, 42, hours=20, now=NOW)
    with sf() as s:
        schedule_reminder(s, 42, hours=20, now=NOW + timedelta(hours=1))
    with sf() as s:
        assert _count(s) == 1  # still one pending reminder, not stacked
        row = s.scalars(select(Reminder)).first()
        assert row.due_at == NOW + timedelta(hours=21)


def test_pop_returns_only_due_and_marks_sent() -> None:
    sf = _factory()
    with sf() as s:
        schedule_reminder(s, 1, hours=20, now=NOW)  # due NOW+20h
        schedule_reminder(s, 2, hours=20, now=NOW - timedelta(hours=30))  # already due
    with sf() as s:
        due = pop_due_reminders(s, now=NOW)
        assert due == [2]
    with sf() as s:
        # The popped one is marked sent and never returns; the other is still pending.
        assert pop_due_reminders(s, now=NOW) == []
        assert pop_due_reminders(s, now=NOW + timedelta(hours=21)) == [1]


def test_pop_empty() -> None:
    sf = _factory()
    with sf() as s:
        assert pop_due_reminders(s, now=NOW) == []
