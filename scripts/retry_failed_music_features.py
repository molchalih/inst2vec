"""CLI wrapper. Real logic lives in modules/music/retry.py."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.config import load_runtime_config
from core.database import init_db
from modules.music.features import MusicSecrets
from modules.music.retry import retry_failed_features


def main() -> None:
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    retry_failed_features(
        music=settings.music,
        paths=settings.paths,
        secrets=MusicSecrets(
            spotify_client_id=secrets.spotify_client_id,
            spotify_client_secret=secrets.spotify_client_secret,
        ),
    )


if __name__ == "__main__":
    main()
