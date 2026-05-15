"""Captions pipeline: clean → detect language → translate.

This __init__ temporarily inlines the legacy implementations from the old
flat modules/captions.py so call sites stay green. Each function is replaced
by a dedicated submodule import in the following tasks.
"""

from __future__ import annotations

from lingua import LanguageDetectorBuilder
from sqlalchemy import func

from modules.captions.state import (
    SCOPE_CLEAN,
    SCOPE_DETECT,
    SCOPE_TRANSLATE,
    clean_caption_text,
)
from modules.console import log, progress
from modules.database import Clip, clip_used_in_analysis, get_session
from modules.external.gemma_translate import GemmaTranslator

__all__ = [
    "clean_captions",
    "detect_caption_language",
    "translate_captions",
]


def clean_captions(commit_every: int) -> None:
    """LEGACY: in-place clean of caption_text. Replaced in Task 5."""
    session = get_session()
    clips = (
        session.query(Clip)
        .filter(
            *clip_used_in_analysis(),
            Clip.caption_text.is_not(None),
            Clip.caption_text != "",
            (Clip.caption_text.contains("@")) | (Clip.caption_text.contains("\n")),
        )
        .order_by(Clip.id)
        .all()
    )
    if not clips:
        session.close()
        return

    cleaned = 0
    for i, clip in enumerate(clips, 1):
        result = clean_caption_text(clip.caption_text)
        if result != clip.caption_text:
            clip.caption_text = result
            cleaned += 1
        if i % commit_every == 0:
            session.commit()

    session.commit()
    session.close()
    log(SCOPE_CLEAN, f"done — {cleaned}/{len(clips)} captions updated")


def detect_caption_language() -> None:
    """LEGACY: detect language on caption_text. Replaced in Task 6."""
    session = get_session()
    clips = (
        session.query(Clip)
        .filter(
            *clip_used_in_analysis(),
            Clip.caption_text.is_not(None),
            Clip.caption_text != "",
            (Clip.caption_language.is_(None)) | (Clip.caption_language == ""),
        )
        .order_by(Clip.id)
        .all()
    )
    if not clips:
        session.close()
        return

    total = len(clips)
    log(SCOPE_DETECT, f"{total} captions to detect")
    detector = LanguageDetectorBuilder.from_all_languages().build()
    detected = 0
    commit_every = 50

    with progress(total, "Detecting languages") as advance:
        for i, clip in enumerate(clips, 1):
            text = (clip.caption_text or "").strip()
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

            if i % commit_every == 0:
                session.commit()

    session.commit()
    session.close()
    log(SCOPE_DETECT, f"done — {detected}/{total} detected", level="ok")


def translate_captions(
    commit_every: int,
    translate_model: str,
    translate_target_lang: str,
    translation_max_chars: int,
    translate_max_new_tokens: int,
) -> None:
    """LEGACY: translate non-English captions. Replaced in Task 7."""
    session = get_session()
    clips = (
        session.query(Clip)
        .filter(
            *clip_used_in_analysis(),
            Clip.caption_text.is_not(None),
            Clip.caption_text != "",
            Clip.caption_language.is_not(None),
            Clip.caption_language != "",
            func.lower(Clip.caption_language).notlike("en%"),
            (Clip.caption_translation.is_(None)) | (Clip.caption_translation == ""),
        )
        .order_by(Clip.id)
        .all()
    )
    if not clips:
        session.close()
        return

    total = len(clips)
    log(SCOPE_TRANSLATE, f"{total} captions to translate")
    translator = GemmaTranslator(model_id=translate_model)
    log(SCOPE_TRANSLATE, f"loading {translator.model_id} on {translator.device}…")
    translated = 0

    with progress(total, "Translating captions") as advance:
        for i, clip in enumerate(clips, 1):
            source = (clip.caption_text or "").strip()[:translation_max_chars]
            source_lang = (clip.caption_language or "").strip().replace("_", "-")
            if not source or not source_lang or source_lang.lower().startswith("en"):
                advance()
                continue

            try:
                translation = translator.translate_text(
                    text=source,
                    source_lang_code=source_lang,
                    target_lang_code=translate_target_lang,
                    max_new_tokens=translate_max_new_tokens,
                )
                if not translation:
                    advance()
                    continue
                clip.caption_translation = translation
                translated += 1
                src_preview = source[:45] + ("…" if len(source) > 45 else "")
                tr_preview = translation[:45] + ("…" if len(translation) > 45 else "")
                advance(detail=f'{clip.id}: "{src_preview}" → "{tr_preview}"')
            except Exception:
                advance()
                continue

            if i % commit_every == 0:
                session.commit()

    session.commit()
    session.close()
    log(SCOPE_TRANSLATE, f"done — {translated}/{total} translated", level="ok")
