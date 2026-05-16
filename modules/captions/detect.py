"""Language detection stage: Lingua → Clip.caption_language from caption_clean."""

from __future__ import annotations

from lingua import LanguageDetectorBuilder
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from modules.captions.state import SCOPE_DETECT
from modules.config import CaptionsSettings
from modules.console import log, progress
from modules.database import Clip, get_engine, needs_caption_language_detection


def detect_caption_language(
    cfg: CaptionsSettings, *, engine: Engine | None = None
) -> None:
    """Tag Clip.caption_language for every clip with a usable caption_clean
    and no language yet. Idempotent: rows with a language stay as-is."""
    eng = engine or get_engine()
    with Session(eng) as session:
        clips = (
            session.query(Clip)
            .filter(*needs_caption_language_detection())
            .order_by(Clip.id)
            .all()
        )
        if not clips:
            return

        total = len(clips)
        log(SCOPE_DETECT, f"{total} captions to detect")
        detector = LanguageDetectorBuilder.from_all_languages().build()
        detected = 0

        with progress(total, "Detecting languages") as advance:
            for i, clip in enumerate(clips, 1):
                text = (clip.caption_clean or "").strip()
                if not text:
                    advance()
                    continue
                lang = detector.detect_language_of(text)
                iso = getattr(lang, "iso_code_639_1", None) if lang else None
                if iso is None:
                    advance()
                    continue
                clip.caption_language = iso.name.lower()
                detected += 1
                advance(detail=f"{clip.id}: {clip.caption_language}")

                if i % cfg.commit_every == 0:
                    session.commit()
        session.commit()
        log(SCOPE_DETECT, f"done — {detected}/{total} detected", level="ok")
