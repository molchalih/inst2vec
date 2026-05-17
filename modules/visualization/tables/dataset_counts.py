"""Small helpers for kept dataset counts."""

from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.database import Clip, User, clip_used_in_analysis

__all__ = ("get_clips_count", "get_users_count")


def get_users_count(eng) -> dict[str, int]:
    with Session(eng) as session:
        all = int(session.query(func.count(User.id)).scalar() or 0)
        kept = int(
            session.query(func.count(User.id))
            .filter(User.is_eligible.is_(True))
            .scalar()
            or 0
        )

        users = {"all": all, "kept": kept}

        return users


def get_clips_count(eng) -> dict[str, int]:
    with Session(eng) as session:
        all = int(session.query(func.count(Clip.id)).scalar() or 0)
        kept = int(
            session.query(func.count(Clip.id)).filter(*clip_used_in_analysis()).scalar()
            or 0
        )

        clips = {"all": all, "kept": kept}

        return clips
