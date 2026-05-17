"""Manual recovery: re-run the captions pipeline (clean → detect → translate)
for any selected+downloaded clip whose caption stage is still unresolved.

Identical decision logic to the main pipeline run — uses the public
``process_captions`` entry point, no duplicated query or model logic.

Usage:
    uv run python scripts/retry_failed_captions.py
"""

from __future__ import annotations

import os
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy.orm import Session  # noqa: E402

from core.config import CaptionsSettings, load_runtime_config  # noqa: E402
from core.console import log  # noqa: E402
from core.database import (  # noqa: E402
    Clip,
    get_engine,
    init_db,
    needs_caption_cleaning,
    needs_caption_language_detection,
    needs_caption_translation,
)
from modules.captions import process_captions  # noqa: E402

SCOPE = "retry-captions"


def retry_failed_captions(cfg: CaptionsSettings) -> None:
    eng = get_engine()
    with Session(eng) as session:
        pending = {
            "clean": session.query(Clip).filter(*needs_caption_cleaning()).count(),
            "detect": session.query(Clip)
            .filter(*needs_caption_language_detection())
            .count(),
            "translate": session.query(Clip)
            .filter(*needs_caption_translation())
            .count(),
        }

    total = sum(pending.values())
    if not total:
        log(SCOPE, "no unresolved caption rows to retry")
        return

    log(
        SCOPE,
        f"retrying captions — clean={pending['clean']} "
        f"detect={pending['detect']} translate={pending['translate']}",
    )
    process_captions(cfg, engine=eng)


if __name__ == "__main__":
    settings, secrets = load_runtime_config()
    init_db(secrets.database_url, secrets.identity_db_url)
    os.makedirs(settings.paths.video_dir, exist_ok=True)
    retry_failed_captions(settings.captions)
