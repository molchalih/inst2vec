"""Manual recovery: re-attempt clips marked is_music_recognized=False.

Single-attempt ACR fingerprint per clip (no internal retries). Small jittered delay
between clips. On match, flips is_music_recognized back to True and links Music row.
Clean no-match leaves the clip at False.

Usage:
    uv run python scripts/retry_failed_music_recognition.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from acrcloud.recognizer import ACRCloudRecognizer

from modules.config import load_runtime_config
from modules.console import log
from modules.database import Clip, clip_used_in_analysis, get_session, init_db
from modules.music.classify import _fingerprint, _get_or_create_music
from modules.services import TransientError

SCOPE = "retry-music"
_DELAY_MIN = 0.3
_DELAY_MAX = 1.5


def retry_failed_music_recognition(
    video_dir: str,
    min_confidence: float,
    arc_host: str,
    arc_access_key: str,
    arc_access_secret: str,
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

        log(SCOPE, f"retrying {len(failed)} failed clips")
        acr = ACRCloudRecognizer(
            {
                "host": arc_host,
                "access_key": arc_access_key,
                "access_secret": arc_access_secret,
                "timeout": 10,
            }
        )
        recovered = 0
        video_dir_path = Path(video_dir)
        for clip in failed:
            time.sleep(random.uniform(_DELAY_MIN, _DELAY_MAX))
            path = video_dir_path / f"{clip.id}.mp4"
            if not path.exists():
                log(SCOPE, f"clip {clip.id}: video missing on disk", level="warn")
                continue
            try:
                result = _fingerprint(acr, str(path), min_confidence, max_attempts=1)
            except TransientError:
                log(SCOPE, f"clip {clip.id} still transient", level="warn")
                continue
            if result:
                artist, track, confidence = result
                music_row = _get_or_create_music(session, artist, track)
                clip.music_id = music_row.id
                clip.music_confidence = confidence
                clip.is_music_recognized = True
                recovered += 1
                session.commit()
                log(SCOPE, f"clip {clip.id}: {artist} – {track} (recovered)")
            else:
                log(SCOPE, f"clip {clip.id}: still no match")

        log(SCOPE, f"recovered {recovered}/{len(failed)}", level="ok")
    finally:
        session.close()


if __name__ == "__main__":
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    os.makedirs(settings.paths.video_dir, exist_ok=True)
    retry_failed_music_recognition(
        video_dir=settings.paths.video_dir,
        min_confidence=settings.music.audio_fingerprint_confidence,
        arc_host=secrets.arc_host,
        arc_access_key=secrets.arc_access_key,
        arc_access_secret=secrets.arc_secret_key,
    )
