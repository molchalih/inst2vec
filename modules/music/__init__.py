"""Music pipeline: classify (ACR) + feature extraction (Spotify/ReccoBeats)."""

from core.config import Secrets, Settings
from modules.music.classify import AcrSecrets, classify_music
from modules.music.features import MusicSecrets, extract_music_features

__all__ = [
    "classify_music",
    "extract_music_features",
    "run_classify",
    "run_features",
]


def run_classify(settings: Settings, secrets: Secrets) -> None:
    """ACR music fingerprinting."""
    classify_music(
        music=settings.music,
        paths=settings.paths,
        secrets=AcrSecrets(
            host=secrets.arc_host,
            access_key=secrets.arc_access_key,
            access_secret=secrets.arc_secret_key,
        ),
    )


def run_features(settings: Settings, secrets: Secrets) -> None:
    """Spotify/ReccoBeats music feature extraction."""
    extract_music_features(
        music=settings.music,
        paths=settings.paths,
        secrets=MusicSecrets(
            spotify_client_id=secrets.spotify_client_id,
            spotify_client_secret=secrets.spotify_client_secret,
        ),
    )
