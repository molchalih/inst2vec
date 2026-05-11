"""Caption language detection and translation pipeline."""

from __future__ import annotations

import os
import re

from lingua import LanguageDetectorBuilder
from sqlalchemy import func, or_

from modules.console import log, progress
from modules.database import Clip, get_session
from modules.external.gemma_translate import GemmaTranslator

COMMIT_EVERY = int(os.environ.get("CAPTIONS_COMMIT_EVERY", 50))
CAPTION_TRANSLATE_MODEL = os.environ.get(
    "CAPTION_TRANSLATE_MODEL", "google/translategemma-4b-it"
)
CAPTION_TRANSLATE_TARGET_LANG = os.environ.get("CAPTION_TRANSLATE_TARGET_LANG", "en")
CAPTION_TRANSLATION_MAX_CHARS = int(
    os.environ.get("CAPTION_TRANSLATION_MAX_CHARS", 1000)
)
CAPTION_TRANSLATE_MAX_NEW_TOKENS = int(
    os.environ.get("CAPTION_TRANSLATE_MAX_NEW_TOKENS", 200)
)

SCOPE_DETECT = "detect_caption_language"
SCOPE_TRANSLATE = "translate_captions"
SCOPE_CLEAN = "clean_captions"

_MENTION_RE = re.compile(r"@[\w.]+")


def _clean(text: str) -> str:
    return " ".join(_MENTION_RE.sub("", text).split())


def clean_captions() -> None:
    """Strip @mentions and collapse whitespace/newlines in caption_text."""
    session = get_session()
    clips = (
        session.query(Clip)
        .filter(
            or_(Clip.disqualified.is_(None), Clip.disqualified == 0),
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
        result = _clean(clip.caption_text)
        if result != clip.caption_text:
            clip.caption_text = result
            cleaned += 1
        if i % COMMIT_EVERY == 0:
            session.commit()

    session.commit()
    session.close()
    log(SCOPE_CLEAN, f"done — {cleaned}/{len(clips)} captions updated")


def detect_caption_language() -> None:
    """Detect caption language with Lingua and write Clip.caption_language."""
    session = get_session()
    clips = (
        session.query(Clip)
        .filter(
            or_(Clip.disqualified.is_(None), Clip.disqualified == 0),
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

            if i % COMMIT_EVERY == 0:
                session.commit()

    session.commit()
    session.close()
    log(SCOPE_DETECT, f"done — {detected}/{total} detected", level="ok")


def translate_captions() -> None:
    """Translate non-English captions with missing translation using TranslateGemma."""
    session = get_session()
    clips = (
        session.query(Clip)
        .filter(
            or_(Clip.disqualified.is_(None), Clip.disqualified == 0),
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
    translator = GemmaTranslator(model_id=CAPTION_TRANSLATE_MODEL)
    log(SCOPE_TRANSLATE, f"loading {translator.model_id} on {translator.device}…")
    translated = 0

    with progress(total, "Translating captions") as advance:
        for i, clip in enumerate(clips, 1):
            source = (clip.caption_text or "").strip()[:CAPTION_TRANSLATION_MAX_CHARS]
            source_lang = (clip.caption_language or "").strip().replace("_", "-")
            if not source or not source_lang or source_lang.lower().startswith("en"):
                advance()
                continue

            try:
                translation = translator.translate_text(
                    text=source,
                    source_lang_code=source_lang,
                    target_lang_code=CAPTION_TRANSLATE_TARGET_LANG,
                    max_new_tokens=CAPTION_TRANSLATE_MAX_NEW_TOKENS,
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

            if i % COMMIT_EVERY == 0:
                session.commit()

    session.commit()
    session.close()
    log(SCOPE_TRANSLATE, f"done — {translated}/{total} translated", level="ok")
