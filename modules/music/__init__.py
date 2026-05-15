"""Music pipeline: classify (ACR) + feature extraction (Spotify/ReccoBeats)."""

from modules.music.classify import classify_music
from modules.music.features import extract_music_features

__all__ = ["classify_music", "extract_music_features"]
