"""Language detection stage: Lingua → Clip.caption_language from caption_clean."""

from __future__ import annotations

import time

from lingua import LanguageDetectorBuilder
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from core.config import CaptionsSettings
from core.console import log, progress
from core.database import Clip, get_engine, needs_caption_language_detection


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
        log("captions:detect", "SCAN", "captions", "ok", stats={"todo": total})
        detector = LanguageDetectorBuilder.from_all_languages().build()
        detected = 0
        t_stage = time.perf_counter()

        with progress(total, "Detecting languages") as advance:
            for i, clip in enumerate(clips, 1):
                text = (clip.caption_clean or "").strip()
                if not text:
                    advance()
                    continue
                t0 = time.perf_counter()
                lang = detector.detect_language_of(text)
                iso = getattr(lang, "iso_code_639_1", None) if lang else None
                if iso is None:
                    log(
                        "captions:detect",
                        "ID",
                        f"cap_{clip.id}",
                        "none",
                        stats={"time": time.perf_counter() - t0},
                    )
                    advance()
                    continue
                clip.caption_language = iso.name.lower()
                detected += 1
                log(
                    "captions:detect",
                    "ID",
                    f"cap_{clip.id}",
                    clip.caption_language,
                    stats={"time": time.perf_counter() - t0},
                )
                advance(detail=f"{clip.id}: {clip.caption_language}")

                if i % cfg.commit_every == 0:
                    session.commit()
        session.commit()
        log(
            "captions:detect",
            "SEAL",
            "detect",
            "ok",
            stats={
                "detected": detected,
                "of": total,
                "time": time.perf_counter() - t_stage,
            },
        )
