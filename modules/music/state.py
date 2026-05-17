"""Shared state constants and helpers for the music package."""

from __future__ import annotations

from sqlalchemy.orm import Session

from modules.database import Clip, Music, clip_used_in_analysis

FEATURE_FIELDS: list[str] = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "valence",
]
UPLOAD_FIELDS: list[str] = [f for f in FEATURE_FIELDS if f not in ("key", "mode")]
_NO_MATCH: str = "none"

SCOPE_CLASSIFY: str = "classify_music"
SCOPE_FEATURES: str = "extract_features"

STAGE_MUSIC_CLASSIFY: str = "music_classify"
STAGE_MUSIC_FEATURES: str = "music_features"
SCOPE_MUSIC: str = "all"


def music_has_features(row: Music) -> bool:
    """True iff every feature column on the row is populated."""
    return all(getattr(row, f) is not None for f in FEATURE_FIELDS)


def reset_music_classify(session: Session) -> None:
    """NULL clip → music links on every eligible clip and delete every
    orphaned Music row (no clip points at it).

    Called on music-classify config drift. The row-level idempotence in
    classify_music (predicate ``Clip.is_music_recognized.is_(None)``) then
    re-fingerprints the cleared clips.
    """
    session.query(Clip).filter(*clip_used_in_analysis()).update(
        {
            Clip.music_id: None,
            Clip.music_confidence: None,
            Clip.is_music_recognized: None,
        },
        synchronize_session=False,
    )
    session.flush()
    # All Music rows are orphans now (no clip points at any of them after
    # the reset above), so a blanket delete of unreferenced Music rows is
    # correct.
    session.query(Music).filter(
        ~Music.id.in_(session.query(Clip.music_id).filter(Clip.music_id.is_not(None)))
    ).delete(synchronize_session=False)
    session.commit()


def reset_music_features(session: Session) -> None:
    """NULL every feature column on every Music row.

    Called on music-features config drift. The row-level idempotence in
    extract_music_features (Spotify/ReccoBeats sub-stages keyed on
    ``spotify_id is None``, ``reccobeats_id is None``,
    ``is_audio_features_extracted is None``) then refills the columns.
    """
    fields: dict = {
        Music.spotify_id: None,
        Music.reccobeats_id: None,
        Music.is_audio_features_extracted: None,
    }
    for f in FEATURE_FIELDS:
        fields[getattr(Music, f)] = None
    session.query(Music).update(fields, synchronize_session=False)
    session.commit()
