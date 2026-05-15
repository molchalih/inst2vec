"""Caption cleaning stage: derive Clip.caption_clean from Clip.caption_text."""

from __future__ import annotations

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from modules.captions.state import SCOPE_CLEAN, clean_caption_text
from modules.config import CaptionsSettings
from modules.console import log
from modules.database import Clip, get_engine, needs_caption_cleaning


def clean_captions(cfg: CaptionsSettings, *, engine: Engine | None = None) -> None:
    """Populate Clip.caption_clean for every selected+downloaded clip with a
    non-empty caption_text and a NULL caption_clean.

    caption_text is never mutated. An empty post-clean result is stored as
    NULL (not "") so the language-detection gate continues to skip the row.
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
        for i, clip in enumerate(clips, 1):
            cleaned = clean_caption_text(clip.caption_text)
            clip.caption_clean = cleaned if cleaned else None
            if cleaned:
                filled += 1
            if i % cfg.commit_every == 0:
                session.commit()
        session.commit()
        log(SCOPE_CLEAN, f"done — {filled}/{total} captions cleaned", level="ok")
