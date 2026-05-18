"""CLI wrapper. Real logic lives in modules/captions/retry.py."""

from __future__ import annotations

import os
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config import load_runtime_config  # noqa: E402
from core.database import init_db  # noqa: E402
from modules.captions.retry import retry_failed_captions  # noqa: E402


def main() -> None:
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    os.makedirs(settings.paths.video_dir, exist_ok=True)
    retry_failed_captions(settings.captions)


if __name__ == "__main__":
    main()
