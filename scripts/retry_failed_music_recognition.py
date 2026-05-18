"""Manual recovery: re-attempt clips marked is_music_recognized=False.

Flips is_music_recognized=False rows back to NULL, then calls public
classify_music with api_max_attempts=1 / acr_max_attempts=1 so each clip
gets a single fresh attempt with no internal retries.

Usage:
    uv run python scripts/retry_failed_music_recognition.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import MusicSettings, PathsSettings, load_runtime_config
from core.console import log
from core.database import Clip, clip_used_in_analysis, get_session, init_db
from modules.music.classify import AcrSecrets, classify_music

SCOPE = "retry-music"


def retry_failed_music_recognition(
    music: MusicSettings,
    paths: PathsSettings,
    secrets: AcrSecrets,
) -> None:
    session = get_session()
    try:
        failed = (
            session.query(Clip)
            .filter(Clip.is_music_recognized.is_(False), *clip_used_in_analysis())
            .all()
        )
        if not failed:
            log(SCOPE, "no failed clips to retry")
            return

        log(SCOPE, f"resetting {len(failed)} failed clips to retryable state")
        for clip in failed:
            clip.is_music_recognized = None
        session.commit()
    finally:
        session.close()

    retry_music = music.model_copy(
        update={
            "api_max_attempts": 1,
            "acr_max_attempts": 1,
            "api_retry_delay": 0.0,
            "api_retry_jitter": 0.0,
        }
    )
    log(SCOPE, "re-running classify_music with single-attempt config")
    classify_music(music=retry_music, paths=paths, secrets=secrets)


if __name__ == "__main__":
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    os.makedirs(settings.paths.video_dir, exist_ok=True)
    retry_failed_music_recognition(
        music=settings.music,
        paths=settings.paths,
        secrets=AcrSecrets(
            host=secrets.arc_host,
            access_key=secrets.arc_access_key,
            access_secret=secrets.arc_secret_key,
        ),
    )
