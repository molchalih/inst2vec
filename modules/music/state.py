"""Shared state constants and helpers for the music package."""

from __future__ import annotations

from modules.database import Music

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


def music_has_features(row: Music) -> bool:
    """True iff every feature column on the row is populated."""
    return all(getattr(row, f) is not None for f in FEATURE_FIELDS)
