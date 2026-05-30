"""DB query helpers for the embeddings pipeline.

Completion is derived from persisted rows in ClipEmbedding/UserEmbedding;
there are no DB status flags for embedding stages.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.database import (
    AudioMIR,
    Clip,
    ClipEmbedding,
    User,
    UserEmbedding,
    clip_used_in_analysis,
)
from modules.embeddings.cases import CASE_REGISTRY


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
    users marked is_eligible."""
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


def get_audio_mir_map(session: Session) -> dict[int, AudioMIR]:
    """Return ``{clip_id: AudioMIR}`` for use by text builders + dependency rows."""
    return {row.clip_id: row for row in session.query(AudioMIR).all()}


_AUDIO_MIR_SIGNATURE_FIELDS: tuple[str, ...] = (
    "is_music_detected",
    "genre_labels",
    "moodtheme_labels",
    "instrument_labels",
    "is_acoustic",
    "is_electronic",
    "is_instrumental",
    "is_happy",
    "is_sad",
    "is_party",
    "is_relaxed",
    "is_aggressive",
    "is_female_voice",
    "is_bright_timbre",
    "is_tonal",
    "danceability",
    "engagement",
    "approachability",
)


def _audio_mir_row_signature(row: AudioMIR | None) -> tuple | None:
    """Deterministic tuple over the AudioMIR fields consumed by verbalize_mir.

    Sentinel for the ``_audio_mir_row`` dependency column: any change to
    one of these fields flips the sandwich / audio per-clip hash, so a
    previously sealed embedding rebuilds.
    """
    if row is None:
        return None
    return tuple(getattr(row, name) for name in _AUDIO_MIR_SIGNATURE_FIELDS)


def dependency_rows_for_case(
    case_name: str,
    clip: Clip,
    *,
    paths,
    audio_mir_map: dict[int, AudioMIR] | None = None,
) -> list[tuple[str, object]]:
    """Return per-clip (column_name, value) pairs for ``case_name``.

    Driven by ``CASE_REGISTRY[case_name].dependency_columns``. The
    synthetic columns ``_video_file_stat`` / ``_audio_file_stat`` stat the
    corresponding file on disk; ``_audio_mir_row`` materializes the
    matching ``AudioMIR`` row's verbalize_mir field signature.

    ``audio_mir_map`` must be supplied whenever a case declares
    ``_audio_mir_row`` in its dependency_columns; the runner / aggregator
    fetches it once per stage run and threads it through.
    """
    case = CASE_REGISTRY[case_name]
    rows: list[tuple[str, object]] = []
    for col in case.dependency_columns:
        if col == "_video_file_stat":
            rows.append((col, fp.file_stat_for_hash(paths.video_for(clip.id))))
        elif col == "_audio_file_stat":
            rows.append((col, fp.file_stat_for_hash(paths.audio_for(clip.id))))
        elif col == "_audio_mir_row":
            if audio_mir_map is None:
                raise ValueError(
                    f"case {case_name!r} declares _audio_mir_row but no "
                    "audio_mir_map was provided to dependency_rows_for_case"
                )
            rows.append((col, _audio_mir_row_signature(audio_mir_map.get(clip.id))))
        else:
            rows.append((col, getattr(clip, col)))
    return rows


def per_clip_source_hashes_and_aggregate(
    session: Session,
    case: str,
    candidate_ids: list[int],
    *,
    settings=None,
) -> tuple[dict[int, str], str]:
    """Return ({clip_id: per_clip_hash}, aggregate_hash) for ``case``."""
    if not candidate_ids:
        return {}, fp.hash_rows([])
    if settings is None:
        raise ValueError(
            "per_clip_source_hashes_and_aggregate requires settings "
            "so paths can be resolved without rereading config.toml"
        )
    paths = settings.paths
    audio_mir_map: dict[int, AudioMIR] | None = None
    if "_audio_mir_row" in CASE_REGISTRY[case].dependency_columns:
        audio_mir_map = get_audio_mir_map(session)
    clips = (
        session.query(Clip).filter(Clip.id.in_(candidate_ids)).order_by(Clip.id).all()
    )
    per_clip: dict[int, str] = {}
    all_rows: list[tuple] = []
    for clip in clips:
        rows = dependency_rows_for_case(
            case, clip, paths=paths, audio_mir_map=audio_mir_map
        )
        per_clip[clip.id] = fp.hash_rows(rows)
        all_rows.extend(rows)
    aggregate = fp.hash_rows(all_rows)
    return per_clip, aggregate


def get_stored_user_hashes(session: Session, case: str) -> dict[int, str | None]:
    """Map user_id → stored source_hash for every UserEmbedding row of ``case``.

    Used by the incremental user-aggregation stage to decide which users
    need recomputing. A row that exists with source_hash=None is treated
    as stale: a previous pre-incremental run wrote it without the hash,
    and we cannot prove it still matches current upstream.
    """
    rows = (
        session.query(UserEmbedding.user_id, UserEmbedding.source_hash)
        .filter(UserEmbedding.embedding_case == case)
        .all()
    )
    return {r.user_id: r.source_hash for r in rows}


def per_user_source_hashes(rows: list[tuple[int, bytes, int]]) -> dict[int, str]:
    """Per-user fingerprint of the (clip_id, blob) pairs the user contributes.

    Accepts the same ``(clip_id, blob, user_id)`` triples consumed by the
    user aggregation; rows are expected to be ordered by ``clip_id`` (the
    aggregation query orders them that way), so the per-user digest is a
    deterministic slice of the stage-level dependency hash. No DB hit.
    """
    by_user: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for clip_id, blob, user_id in rows:
        by_user[user_id].append((clip_id, hashlib.sha256(blob).hexdigest()))
    return {user_id: fp.hash_rows(pairs) for user_id, pairs in by_user.items()}
