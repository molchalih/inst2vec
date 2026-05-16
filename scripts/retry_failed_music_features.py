"""Manual recovery: re-attempt Music rows marked is_audio_features_extracted=False.

Resets terminal markers on target rows (is_audio_features_extracted=NULL;
spotify_id/reccobeats_id from _NO_MATCH back to NULL) and re-runs
extract_music_features with api_max_attempts=1 — a single fresh attempt per
call site, no internal retries.

Usage:
    uv run python scripts/retry_failed_music_features.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.config import MusicSettings, PathsSettings, load_runtime_config
from modules.console import log
from modules.database import Music, get_session, init_db
from modules.music.features import MusicSecrets, extract_music_features
from modules.music.state import _NO_MATCH

SCOPE = "retry-features"


def retry_failed_music_features(
    music: MusicSettings,
    paths: PathsSettings,
    secrets: MusicSecrets,
) -> None:
    session = get_session()
    try:
        failed = (
            session.query(Music)
            .filter(Music.is_audio_features_extracted.is_(False))
            .all()
        )
        if not failed:
            log(SCOPE, "no failed music rows to retry")
            return

        log(SCOPE, f"resetting {len(failed)} failed music rows")
        for row in failed:
            row.is_audio_features_extracted = None
            if row.spotify_id == _NO_MATCH:
                row.spotify_id = None
            if row.reccobeats_id == _NO_MATCH:
                row.reccobeats_id = None
        session.commit()
    finally:
        session.close()

    retry_music = music.model_copy(
        update={
            "api_max_attempts": 1,
            "api_retry_delay": 0.0,
            "api_retry_jitter": 0.0,
        }
    )
    log(SCOPE, "re-running extract_music_features with api_max_attempts=1")
    extract_music_features(music=retry_music, paths=paths, secrets=secrets)


if __name__ == "__main__":
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    retry_failed_music_features(
        music=settings.music,
        paths=settings.paths,
        secrets=MusicSecrets(
            spotify_client_id=secrets.spotify_client_id,
            spotify_client_secret=secrets.spotify_client_secret,
        ),
    )
