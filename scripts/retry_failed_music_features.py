"""Manual recovery: re-attempt Music rows marked is_audio_features_extracted=False.

Resets terminal markers on target rows (is_audio_features_extracted=NULL;
recognition_status from "no_match" back to "pending") and re-runs
extract_music_features with api_max_attempts=1 — a single fresh attempt per
call site, no internal retries. The reccobeats stage auto-retries rows
with reccobeats_id IS NULL, so no explicit reccobeats reset is needed.

Usage:
    uv run python scripts/retry_failed_music_features.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import MusicSettings, PathsSettings, load_runtime_config
from core.console import log
from core.database import Music, get_session, init_db
from modules.music.features import MusicSecrets, extract_music_features

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

        rows = session.query(Music).filter(Music.recognition_status == "no_match").all()
        for row in rows:
            row.recognition_status = "pending"
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
