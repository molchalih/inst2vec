"""Caption cleaning stage: derive Clip.caption_clean from Clip.caption_text."""

from __future__ import annotations

import time

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from core.config import CaptionsSettings
from core.console import log
from core.database import Clip, get_engine, needs_caption_cleaning
from modules.captions.state import clean_caption_text


def clean_captions(cfg: CaptionsSettings, *, engine: Engine | None = None) -> None:
    """Populate Clip.caption_clean for every selected+downloaded clip with a
    non-empty caption_text and a NULL caption_clean.

    caption_text is never mutated. An empty post-clean result is stored as ""
    (empty string) to mark the row as processed; NULL means not yet cleaned.
    Downstream predicates gate on both is_not(None) and trim != "", so ""
    rows are correctly excluded from language detection and translation.
    """
    eng = engine or get_engine()
    with Session(eng) as session:
        clips = (
            session.query(Clip)
            .filter(*needs_caption_cleaning())
            .order_by(Clip.id)
            .all()
        )
        if not clips:
            return

        total = len(clips)
        filled = 0
        t_stage = time.perf_counter()
        for i, clip in enumerate(clips, 1):
            cleaned = clean_caption_text(clip.caption_text)
            clip.caption_clean = cleaned if cleaned else ""
            if cleaned:
                filled += 1
            if i % cfg.commit_every == 0:
                session.commit()
        session.commit()
        log(
            "captions:clean",
            "CLEAN",
            "captions",
            "ok",
            stats={
                "in": total,
                "filled": filled,
                "time": time.perf_counter() - t_stage,
            },
        )
