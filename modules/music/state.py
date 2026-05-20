"""Shared state constants and helpers for the music package."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from core.config import MusicSettings
from core.database import Clip, Music
from core.pipeline import Stage

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

STAGE_MUSIC_CLASSIFY: Stage = Stage.MUSIC_CLASSIFY
STAGE_MUSIC_FEATURES: Stage = Stage.MUSIC_FEATURES
SCOPE_MUSIC: str = "all"

# Fields whose values can change the *outputs* of each music stage.
# Purely-operational knobs (commit_every, retry attempts/delays/jitter)
# are intentionally excluded so a value change does not invalidate the
# stored fingerprint.
_CLASSIFY_CONFIG_FIELDS: tuple[str, ...] = ("audio_fingerprint_confidence",)
_FEATURES_CONFIG_FIELDS: tuple[str, ...] = (
    "http_timeout",
    "spotify_search_limit",
    "spotify_token_skew_seconds",
    "spotify_request_timeout",
    "reccobeats_batch_size",
    "reccobeats_delay_min",
    "reccobeats_delay_max",
    "manual_features_max_seconds",
    "manual_features_sample_rate",
    "manual_features_max_mb",
    "manual_features_mp3_bitrate",
    "ffmpeg_timeout_seconds",
)


def _stable_subset_payload(music: MusicSettings, fields: tuple[str, ...]) -> str:
    payload = {f: getattr(music, f) for f in fields}
    return json.dumps(payload, sort_keys=True, default=str)


def classify_config_payload(music: MusicSettings) -> str:
    """Stable JSON of the MusicSettings fields that affect classify outputs."""
    return _stable_subset_payload(music, _CLASSIFY_CONFIG_FIELDS)


def features_config_payload(music: MusicSettings) -> str:
    """Stable JSON of the MusicSettings fields that affect feature outputs."""
    return _stable_subset_payload(music, _FEATURES_CONFIG_FIELDS)


def music_has_features(row: Music) -> bool:
    """True iff every feature column on the row is populated."""
    return all(getattr(row, f) is not None for f in FEATURE_FIELDS)


def reset_music_classify(session: Session) -> None:
    """NULL clip → music links on every clip and delete every orphaned
    Music row.

    Called on music-classify config drift. Resets all clips (not just
    currently-eligible ones) so clips that re-enter the selection pool
    on a later run can't carry stale ``is_music_recognized``/``music_id``
    values produced under the previous config. The row-level idempotence
    in classify_music (predicate ``Clip.is_music_recognized.is_(None)``)
    re-fingerprints the cleared eligible clips on the next pass.
    """
    session.query(Clip).update(
        {
            Clip.music_id: None,
            Clip.music_confidence: None,
            Clip.is_music_recognized: None,
        },
        synchronize_session=False,
    )
    session.flush()
    # No clip points at any Music row after the reset above, so every
    # Music row is now an orphan and a blanket delete is correct.
    session.query(Music).delete(synchronize_session=False)
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
        Music.is_reccobeats_resolved: None,
        Music.is_audio_features_extracted: None,
        Music.recognition_status: "pending",
    }
    for f in FEATURE_FIELDS:
        fields[getattr(Music, f)] = None
    session.query(Music).update(fields, synchronize_session=False)
    session.commit()
