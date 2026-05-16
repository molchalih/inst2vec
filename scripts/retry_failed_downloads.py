"""Manual recovery: re-attempt clips marked is_downloaded = False.

No automatic retries (max_attempts = 1). Small random delay between downloads.
On success, flips Clip.is_downloaded back to True.

Usage:
    uv run python scripts/retry_failed_downloads.py
"""

from __future__ import annotations

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.config import load_runtime_config
from modules.console import log
from modules.database import Clip, get_session, init_db
from modules.download import fetch_file

SCOPE = "retry"
_DELAY_MIN = 0.3
_DELAY_MAX = 1.5


def retry_failed_downloads(video_dir: str) -> None:
    os.makedirs(video_dir, exist_ok=True)

    session = get_session()
    try:
        failed = (
            session.query(Clip)
            .filter(Clip.is_selected.is_(True), Clip.is_downloaded.is_(False))
            .all()
        )
        if not failed:
            log(SCOPE, "no failed clips to retry")
            return

        log(SCOPE, f"retrying {len(failed)} failed clips")
        recovered = 0
        for clip in failed:
            time.sleep(random.uniform(_DELAY_MIN, _DELAY_MAX))
            if clip.video_url is None:
                log(SCOPE, f"clip {clip.id}: no video_url, skipping", level="warn")
                continue
            path = os.path.join(video_dir, f"{clip.id}.mp4")
            ok = fetch_file(
                clip.video_url,
                path,
                max_attempts=1,
                retry_delay=0,
                retry_jitter=0,
            )
            if ok:
                clip.is_downloaded = True
                recovered += 1
                session.commit()
            else:
                log(SCOPE, f"clip {clip.id} still failed", level="warn")

        log(SCOPE, f"recovered {recovered}/{len(failed)}")
    finally:
        session.close()


if __name__ == "__main__":
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    retry_failed_downloads(video_dir=settings.paths.video_dir)
