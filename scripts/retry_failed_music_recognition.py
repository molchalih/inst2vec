"""CLI wrapper. Real logic lives in modules/music/retry.py."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import load_runtime_config
from core.database import init_db
from modules.music.classify import AcrSecrets
from modules.music.retry import retry_failed_recognition


def main() -> None:
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    os.makedirs(settings.paths.video_dir, exist_ok=True)
    retry_failed_recognition(
        music=settings.music,
        paths=settings.paths,
        secrets=AcrSecrets(
            host=secrets.arc_host,
            access_key=secrets.arc_access_key,
            access_secret=secrets.arc_secret_key,
        ),
    )


if __name__ == "__main__":
    main()
