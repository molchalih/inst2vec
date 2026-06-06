"""Schedule + drain "come back later" reminders (pure over a Session).

The Mini App's rest-nudge lets a tired annotator ask to be pinged later; this
records one pending reminder per telegram id and hands due ones to the bot.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from swipe_anchor.db.models import Reminder


def schedule_reminder(
    session: Session,
    telegram_id: int,
    *,
    hours: float,
    now: datetime | None = None,
) -> datetime:
    """Upsert the single pending reminder for this user; return its due time.

    Re-tapping just pushes the existing pending reminder out — never stacks up
    multiple DMs for one person.
    """
    now = now or datetime.now(UTC)
    due = now + timedelta(hours=hours)
    pending = session.scalars(
        select(Reminder).where(
            Reminder.telegram_id == telegram_id, Reminder.sent.is_(False)
        )
    ).first()
    if pending is None:
        session.add(Reminder(telegram_id=telegram_id, due_at=due, sent=False))
    else:
        pending.due_at = due
    return due


def pop_due_reminders(
    session: Session, *, now: datetime | None = None
) -> list[int]:
    """Mark every due, unsent reminder as sent and return their telegram ids."""
    now = now or datetime.now(UTC)
    due = session.scalars(
        select(Reminder).where(Reminder.sent.is_(False), Reminder.due_at <= now)
    ).all()
    ids: list[int] = []
    for row in due:
        row.sent = True
        ids.append(row.telegram_id)
    return ids
