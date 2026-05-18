"""Manual recovery for failed Captions rows.

Functions here are also invoked by ``scripts/retry_failed_captions.py``
CLI wrapper. Tests import from this module directly.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from core.config import CaptionsSettings
from core.console import log
from core.database import (
    Clip,
    get_engine,
    needs_caption_cleaning,
    needs_caption_language_detection,
    needs_caption_translation,
)
from modules.captions import process_captions

SCOPE = "retry-captions"


def retry_failed_captions(cfg: CaptionsSettings) -> None:
    """Re-run the captions pipeline for any unresolved caption rows.

    Covers the clean → detect → translate stages. Uses the public
    process_captions entry point, no duplicated query or model logic.
    """
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
