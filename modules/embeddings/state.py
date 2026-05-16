"""DB query helpers for the embeddings pipeline.

Completion is derived from persisted rows in ClipEmbedding/UserEmbedding;
there are no DB status flags for embedding stages.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from modules import fingerprint as fp
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


def get_embedded_source_hashes(session: Session, case: str) -> dict[int, str | None]:
    """Map clip_id → stored source_hash for every ClipEmbedding row of ``case``.

    Used by the incremental runner to decide which clips need re-embedding.
    A row that exists with source_hash=None is treated as stale: a previous
    pre-incremental run wrote it without the hash, and we cannot prove it
    still matches current upstream.
    """
    rows = (
        session.query(ClipEmbedding.clip_id, ClipEmbedding.source_hash)
        .filter(ClipEmbedding.embedding_case == case)
        .all()
    )
    return {r.clip_id: r.source_hash for r in rows}


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
    session: Session, case: str, exclude_disqualified_users: bool
) -> list[tuple[int, bytes, int]]:
    """Return (clip_id, embedding_blob, user_id) rows for the given case.

    Filters out clips that are no longer in the candidate set so orphan
    rows (clips later deselected, undownloaded, or belonging to users
    flipped to ``is_eligible=False`` when ``exclude_disqualified_users``
    is True) do not contaminate user means.

    The candidate filter mirrors ``get_clip_embedding_candidates`` so the
    user-embedding stage and the clip-embedding stage agree on what
    counts as eligible. Results are ordered by clip_id so callers can
    derive a deterministic fingerprint from the same rows.
    """
    q = (
        session.query(ClipEmbedding.clip_id, ClipEmbedding.embedding, Clip.user_id)
        .join(Clip, ClipEmbedding.clip_id == Clip.id)
        .filter(
            ClipEmbedding.embedding_case == case,
            *clip_used_in_analysis(),
        )
    )
    if exclude_disqualified_users:
        q = q.join(User, Clip.user_id == User.id).filter(User.is_eligible.is_(True))
    return q.order_by(ClipEmbedding.clip_id).all()


def get_music_map(session: Session) -> dict[int, Music]:
    """Return {music_id: Music} for use by text builders that verbalize music."""
    return {m.id: m for m in session.query(Music).all()}


def dependency_rows_for_case(
    session: Session, case: str, candidate_ids: list[int]
) -> list[tuple]:
    """Return the per-candidate tuple of upstream output state for ``case``.

    The columns selected mirror what the case's payload_builder and
    text_builder actually read; the result is sorted by clip_id so the
    digest produced by ``fingerprint.hash_rows`` is deterministic.

    video    : (id, is_downloaded)
    sandwich : (id, is_downloaded, music_id, caption_*, speech_*,
                Music.energy/valence/acousticness/instrumentalness/
                danceability/speechiness/tempo/mode/key/track/artist)
    audio    : (id,                music_id, speech_*,
                Music.energy/...) — captions deliberately excluded
                (build_audio_text does not read them).
    """
    if not candidate_ids:
        return []

    if case == "video":
        rows = (
            session.query(Clip.id, Clip.is_downloaded)
            .filter(Clip.id.in_(candidate_ids))
            .order_by(Clip.id)
            .all()
        )
        return [tuple(r) for r in rows]

    music_cols = (
        Music.energy,
        Music.valence,
        Music.acousticness,
        Music.instrumentalness,
        Music.danceability,
        Music.speechiness,
        Music.tempo,
        Music.mode,
        Music.key,
        Music.track,
        Music.artist,
    )

    if case == "sandwich":
        rows = (
            session.query(
                Clip.id,
                Clip.is_downloaded,
                Clip.music_id,
                Clip.caption_text,
                Clip.caption_clean,
                Clip.caption_language,
                Clip.caption_translation,
                Clip.speech_transcription,
                Clip.speech_language,
                Clip.speech_translation,
                *music_cols,
            )
            .outerjoin(Music, Clip.music_id == Music.id)
            .filter(Clip.id.in_(candidate_ids))
            .order_by(Clip.id)
            .all()
        )
        return [tuple(r) for r in rows]

    if case == "audio":
        rows = (
            session.query(
                Clip.id,
                Clip.music_id,
                Clip.speech_transcription,
                Clip.speech_language,
                Clip.speech_translation,
                *music_cols,
            )
            .outerjoin(Music, Clip.music_id == Music.id)
            .filter(Clip.id.in_(candidate_ids))
            .order_by(Clip.id)
            .all()
        )
        return [tuple(r) for r in rows]

    raise ValueError(f"Unknown embedding case: {case!r}")


def per_clip_source_hashes_and_aggregate(
    session: Session, case: str, candidate_ids: list[int]
) -> tuple[dict[int, str], str]:
    """Return ({clip_id: per_clip_hash}, aggregate_hash) for ``case``.

    Both values are derived from the same call to ``dependency_rows_for_case``
    so the per-clip hashes and the stage-level aggregate stay byte-identical.
    """
    rows = dependency_rows_for_case(session, case, candidate_ids)
    per_clip = {r[0]: fp.hash_rows([r]) for r in rows}
    aggregate = fp.hash_rows(rows)
    return per_clip, aggregate
