"""DB query helpers for the embeddings pipeline.

Completion is derived from persisted rows in ClipEmbedding/UserEmbedding;
there are no DB status flags for embedding stages.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from modules.database import (
    Clip,
    ClipEmbedding,
    Music,
    User,
    UserEmbedding,
    clip_used_in_analysis,
)


def get_embedded_clip_ids(session: Session, case: str) -> set[int]:
    """Clip ids that already have a ClipEmbedding row for ``case``."""
    rows = (
        session.query(ClipEmbedding.clip_id)
        .filter(ClipEmbedding.embedding_case == case)
        .all()
    )
    return {r.clip_id for r in rows}


def get_embedded_user_ids(session: Session, case: str) -> set[int]:
    """User ids that already have a UserEmbedding row for ``case``."""
    rows = (
        session.query(UserEmbedding.user_id)
        .filter(UserEmbedding.embedding_case == case)
        .all()
    )
    return {r.user_id for r in rows}


def get_clip_embedding_candidates(
    session: Session, exclude_disqualified_users: bool
) -> list[Clip]:
    """Eligible clips (selected + downloaded), optionally restricted to
    users marked is_eligible.
    """
    q = session.query(Clip).filter(*clip_used_in_analysis())
    if exclude_disqualified_users:
        q = q.join(User, Clip.user_id == User.id).filter(User.is_eligible.is_(True))
    return q.all()


def get_clip_embedding_rows_for_user_aggregation(
    session: Session, case: str
) -> list[tuple[bytes, int]]:
    """Return (embedding_blob, user_id) rows for the given case."""
    return (
        session.query(ClipEmbedding.embedding, Clip.user_id)
        .join(Clip, ClipEmbedding.clip_id == Clip.id)
        .filter(ClipEmbedding.embedding_case == case)
        .all()
    )


def get_music_map(session: Session) -> dict[int, Music]:
    """Return {music_id: Music} for use by text builders that verbalize music."""
    return {m.id: m for m in session.query(Music).all()}
