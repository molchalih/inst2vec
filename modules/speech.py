"""Phase 2 – speech pipeline.

classify_speech()  Whisper transcription → has_speech + speech_confidence.
translate_speech() TranslateGemma        → speech_translation.
"""

from __future__ import annotations

import os
import re

import whisper
from sqlalchemy import func, or_

from modules.console import log, progress
from modules.database import Clip, get_session
from modules.external.gemma_translate import GemmaTranslator

VIDEO_DIR = os.environ.get("VIDEO_DIR", "data/source/videos")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
COMMIT_EVERY = int(os.environ.get("SPEECH_COMMIT_EVERY", 50))
SPEECH_TRANSLATE_MODEL = os.environ.get(
    "SPEECH_TRANSLATE_MODEL", "google/translategemma-4b-it"
)
SPEECH_TRANSLATE_TARGET_LANG = os.environ.get("SPEECH_TRANSLATE_TARGET_LANG", "en")
SPEECH_TRANSLATION_MAX_CHARS = int(os.environ.get("SPEECH_TRANSLATION_MAX_CHARS", 1000))
SPEECH_TRANSLATE_MAX_NEW_TOKENS = int(
    os.environ.get("SPEECH_TRANSLATE_MAX_NEW_TOKENS", 200)
)
# Two complementary hallucination gates (both configurable via .env):
#
#   SPEECH_LOGPROB_THRESHOLD    — mean avg_logprob across segments.
#                                 Genuine speech is typically > -0.5; hallucinations on
#                                 music/silence fall in -0.8 to -1.5+. Fires on uncertain text.
#
#   SPEECH_COMPRESSION_THRESHOLD — mean compression_ratio across segments.
#                                 Whisper default ceiling is 2.4; repetitive loops like
#                                 "Thank you. Thank you." score high even with low logprob,
#                                 so this catches confident-but-repetitive hallucinations.
#
# A clip is marked has_speech=0 if EITHER gate fires. Text is always stored regardless.
LOGPROB_THRESHOLD = float(os.environ.get("SPEECH_LOGPROB_THRESHOLD", "-0.8"))
COMPRESSION_THRESHOLD = float(os.environ.get("SPEECH_COMPRESSION_THRESHOLD", "2.4"))
SPEECH_MIN_MEANINGFUL_CHARS = int(os.environ.get("SPEECH_MIN_MEANINGFUL_CHARS", "8"))

SCOPE_CLASSIFY = "classify_speech"
SCOPE_TRANSLATE = "translate_speech"
SCOPE_CLEAN = "clean_speech"

# Substrings in speech_translation that indicate a hallucination / bad transcription.
# Any clip whose translation matches one of these will have has_speech reset to 0.
_HALLUCINATION_MARKERS = [
    "DimaTorzok",
]

_NON_LETTER_RE = re.compile(r"[^A-Za-zА-Яа-я\u00C0-\u024F\u0400-\u04FF]+")


def _transcribe(model, path: str) -> tuple[str, str, float, float, float]:
    """Return (text, language, speech_confidence, avg_logprob, compression_ratio).

    language             = BCP-47 code detected by Whisper (e.g. "en", "ru").
    speech_confidence    = 1 - mean(no_speech_prob) — how likely the model heard speech.
    avg_logprob          = mean per-token log-probability — how certain the output tokens are.
    compression_ratio    = mean gzip ratio of token sequences — high = repetitive output.
    All float metrics are 0.0 when Whisper produces no segments.
    """
    result = model.transcribe(path)
    text = (result.get("text") or "").strip()
    language = result.get("language") or ""
    segs = result.get("segments") or []
    if segs:
        mean_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segs) / len(segs)
        confidence = max(0.0, min(1.0, 1.0 - mean_no_speech))
        avg_logprob = sum(s.get("avg_logprob", 0.0) for s in segs) / len(segs)
        compression_ratio = sum(s.get("compression_ratio", 0.0) for s in segs) / len(
            segs
        )
    else:
        confidence = avg_logprob = compression_ratio = 0.0
    return text, language, confidence, avg_logprob, compression_ratio


def _has_meaningful_speech_text(text: str) -> bool:
    cleaned = _NON_LETTER_RE.sub("", (text or "").strip())
    return len(cleaned) >= SPEECH_MIN_MEANINGFUL_CHARS


# ── public API ─────────────────────────────────────────────────────────────────


def clean_speech() -> None:
    """Reset has_speech to 0 for clips whose speech_translation contains hallucination markers."""
    session = get_session()
    filter_conditions = [
        Clip.speech_translation.contains(marker) for marker in _HALLUCINATION_MARKERS
    ]
    clips = (
        session.query(Clip)
        .filter(
            or_(Clip.disqualified.is_(None), Clip.disqualified == 0),
            Clip.has_speech == 1,
            Clip.speech_translation.is_not(None),
            or_(*filter_conditions),
        )
        .order_by(Clip.pk)
        .all()
    )
    if not clips:
        session.close()
        return

    for clip in clips:
        clip.has_speech = 0
        log(
            SCOPE_CLEAN,
            f'{clip.pk}: marked has_speech=0 (translation: "{(clip.speech_translation or "")[:60]}")',
        )

    session.commit()
    session.close()
    log(SCOPE_CLEAN, f"done — {len(clips)} clips cleared")


def classify_speech() -> None:
    """Transcribe all unresolved clips with Whisper, tagging has_speech and speech_confidence.

    Clips with missing video files are skipped and retried next run.
    """
    session = get_session()
    clips = (
        session.query(Clip)
        .filter(
            Clip.has_speech.is_(None),
            or_(Clip.disqualified.is_(None), Clip.disqualified == 0),
        )
        .order_by(Clip.pk.desc())
        .all()
    )
    if not clips:
        session.close()
        return

    log(SCOPE_CLASSIFY, f"{len(clips)} clips to transcribe")
    model: whisper.Whisper | None = None
    has_speech = no_speech = missing = 0

    with progress(len(clips), "Transcribing") as advance:
        for i, clip in enumerate(clips, 1):
            path = f"{VIDEO_DIR}/{clip.pk}.mp4"
            if not os.path.exists(path):
                missing += 1
                advance()
                continue

            if model is None:
                log(SCOPE_CLASSIFY, f"loading {WHISPER_MODEL}…")
                model = whisper.load_model(WHISPER_MODEL)
            try:
                text, language, conf, avg_logprob, compression_ratio = _transcribe(
                    model, path
                )
            except Exception:
                text, language, conf, avg_logprob, compression_ratio = (
                    "",
                    "",
                    0.0,
                    0.0,
                    0.0,
                )

            clip.speech_transcription = text
            clip.speech_language = language or None
            clip.speech_confidence = conf if text else None
            clip.speech_avg_logprob = avg_logprob if text else None
            clip.speech_compression_ratio = compression_ratio if text else None

            low_logprob = bool(text) and avg_logprob < LOGPROB_THRESHOLD
            high_compression = bool(text) and compression_ratio > COMPRESSION_THRESHOLD
            hallucination = low_logprob or high_compression
            meaningful = _has_meaningful_speech_text(text)

            if meaningful and not hallucination:
                clip.has_speech = 1
                has_speech += 1
                preview = text[:60] + ("…" if len(text) > 60 else "")
                advance(detail=f'{clip.pk}: "{preview}"')
            else:
                clip.has_speech = 0
                no_speech += 1
                advance()

            if i % COMMIT_EVERY == 0:
                session.commit()

    session.commit()
    session.close()
    parts = [f"{has_speech} with speech", f"{no_speech} silent"]
    if missing:
        parts.append(f"{missing} skipped (video not downloaded yet)")
    log(SCOPE_CLASSIFY, f"done — {', '.join(parts)}", level="ok")


def translate_speech() -> None:
    """Translate all non-empty transcriptions with missing translation using TranslateGemma."""
    session = get_session()
    clips = (
        session.query(Clip)
        .filter(
            or_(Clip.disqualified.is_(None), Clip.disqualified == 0),
            Clip.speech_transcription.is_not(None),
            Clip.speech_transcription != "",
            Clip.speech_language.is_not(None),
            Clip.speech_language != "",
            func.lower(Clip.speech_language).notlike("en%"),
            (Clip.speech_translation.is_(None)) | (Clip.speech_translation == ""),
        )
        .order_by(Clip.pk)
        .all()
    )
    if not clips:
        session.close()
        return

    total = len(clips)
    log(SCOPE_TRANSLATE, f"{total} clips to translate")
    translator = GemmaTranslator(model_id=SPEECH_TRANSLATE_MODEL)
    log(SCOPE_TRANSLATE, f"loading {translator.model_id} on {translator.device}…")
    translated = 0

    with progress(total, "Translating speech") as advance:
        for i, clip in enumerate(clips, 1):
            source = (clip.speech_transcription or "").strip()[
                :SPEECH_TRANSLATION_MAX_CHARS
            ]
            source_lang = (clip.speech_language or "").strip().replace("_", "-")
            if not source or not source_lang or source_lang.lower().startswith("en"):
                advance()
                continue

            try:
                translation = translator.translate_text(
                    text=source,
                    source_lang_code=source_lang,
                    target_lang_code=SPEECH_TRANSLATE_TARGET_LANG,
                    max_new_tokens=SPEECH_TRANSLATE_MAX_NEW_TOKENS,
                )
                if not translation:
                    advance()
                    continue
                clip.speech_translation = translation
                translated += 1
                src_preview = source[:45] + ("…" if len(source) > 45 else "")
                tr_preview = translation[:45] + ("…" if len(translation) > 45 else "")
                advance(detail=f'{clip.pk}: "{src_preview}" → "{tr_preview}"')
            except Exception:
                advance()
                continue

            if i % COMMIT_EVERY == 0:
                session.commit()

    session.commit()
    session.close()
    log(SCOPE_TRANSLATE, f"done — {translated}/{total} translated", level="ok")
