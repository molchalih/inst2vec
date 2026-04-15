"""Phase 2 – speech pipeline.

  classify_speech()  Whisper transcription → has_speech + speech_confidence.
  translate_speech() TranslateGemma        → speech_translation.
"""
from __future__ import annotations

import os
from typing import Optional

import whisper
from sqlalchemy import func

from modules.database import Clip, get_session
from modules.external.gemma_translate import GemmaTranslator
from modules.services import log

VIDEO_DIR = os.environ.get("VIDEO_DIR", "data/source/videos")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
COMMIT_EVERY = int(os.environ.get("SPEECH_COMMIT_EVERY", 50))
SPEECH_TRANSLATE_MODEL = os.environ.get("SPEECH_TRANSLATE_MODEL", "google/translategemma-4b-it")
SPEECH_TRANSLATE_TARGET_LANG = os.environ.get("SPEECH_TRANSLATE_TARGET_LANG", "en")
SPEECH_TRANSLATION_MAX_CHARS = int(os.environ.get("SPEECH_TRANSLATION_MAX_CHARS", 1000))
SPEECH_TRANSLATE_MAX_NEW_TOKENS = int(os.environ.get("SPEECH_TRANSLATE_MAX_NEW_TOKENS", 200))
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

SCOPE_CLASSIFY = "classify_speech"
SCOPE_TRANSLATE = "translate_speech"


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
        compression_ratio = sum(s.get("compression_ratio", 0.0) for s in segs) / len(segs)
    else:
        confidence = avg_logprob = compression_ratio = 0.0
    return text, language, confidence, avg_logprob, compression_ratio


# ── public API ─────────────────────────────────────────────────────────────────

def classify_speech() -> None:
    """Transcribe all unresolved clips with Whisper, tagging has_speech and speech_confidence.

    Clips that already have speech_transcription from a prior run get their flags resolved
    without re-running Whisper. Clips with missing video files are skipped and retried next run.
    """
    session = get_session()
    clips = session.query(Clip).filter(Clip.has_speech.is_(None)).order_by(Clip.pk.desc()).all()
    if not clips:
        session.close()
        return

    log(SCOPE_CLASSIFY, f"{len(clips)} clips to transcribe")
    model: Optional[whisper.Whisper] = None
    has_speech = no_speech = missing = 0

    for i, clip in enumerate(clips, 1):
        path = f"{VIDEO_DIR}/{clip.pk}.mp4"
        if not os.path.exists(path):
            missing += 1
            continue

        if clip.speech_transcription is not None:
            # flags not yet resolved but transcription already stored from a prior run;
            # avg_logprob unavailable for legacy data — leave NULL, resolve flag only
            has = 1 if clip.speech_transcription.strip() else 0
            clip.has_speech = has
            if has and clip.speech_confidence is None:
                clip.speech_confidence = 1.0
            has_speech += has
            no_speech += 1 - has
        else:
            if model is None:
                log(SCOPE_CLASSIFY, f"loading {WHISPER_MODEL}…")
                model = whisper.load_model(WHISPER_MODEL)
            try:
                text, language, conf, avg_logprob, compression_ratio = _transcribe(model, path)
            except Exception:
                text, language, conf, avg_logprob, compression_ratio = "", "", 0.0, 0.0, 0.0

            # Always persist transcription and all quality metrics regardless of gate result.
            clip.speech_transcription = text
            clip.speech_language = language or None
            clip.speech_confidence = conf if text else None
            clip.speech_avg_logprob = avg_logprob if text else None
            clip.speech_compression_ratio = compression_ratio if text else None

            low_logprob = bool(text) and avg_logprob < LOGPROB_THRESHOLD
            high_compression = bool(text) and compression_ratio > COMPRESSION_THRESHOLD
            hallucination = low_logprob or high_compression

            if text and not hallucination:
                clip.has_speech = 1
                has_speech += 1
                preview = text[:60] + ("…" if len(text) > 60 else "")
                log(SCOPE_CLASSIFY, f"{i}/{len(clips)} — {clip.pk}: \"{preview}\" (logprob {avg_logprob:.2f}, ratio {compression_ratio:.2f})")
            else:
                clip.has_speech = 0
                no_speech += 1
                if hallucination:
                    reason = []
                    if low_logprob:
                        reason.append(f"logprob {avg_logprob:.2f}")
                    if high_compression:
                        reason.append(f"ratio {compression_ratio:.2f}")
                    preview = text[:50] + ("…" if len(text) > 50 else "")
                    log(SCOPE_CLASSIFY, f"{i}/{len(clips)} — {clip.pk}: hallucination [{', '.join(reason)}] \"{preview}\"")

        if i % COMMIT_EVERY == 0:
            session.commit()

    session.commit()
    session.close()
    parts = [f"{has_speech} with speech", f"{no_speech} silent"]
    if missing:
        parts.append(f"{missing} skipped (video not downloaded yet)")
    log(SCOPE_CLASSIFY, f"done — {', '.join(parts)}")


def translate_speech() -> None:
    """Translate all non-empty transcriptions with missing translation using TranslateGemma."""
    session = get_session()
    clips = (
        session.query(Clip)
        .filter(
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

    for i, clip in enumerate(clips, 1):
        source = (clip.speech_transcription or "").strip()[:SPEECH_TRANSLATION_MAX_CHARS]
        source_lang = (clip.speech_language or "").strip().replace("_", "-")
        if not source:
            continue
        if not source_lang or source_lang.lower().startswith("en"):
            continue

        try:
            translation = translator.translate_text(
                text=source,
                source_lang_code=source_lang,
                target_lang_code=SPEECH_TRANSLATE_TARGET_LANG,
                max_new_tokens=SPEECH_TRANSLATE_MAX_NEW_TOKENS,
            )
            if not translation:
                continue
            clip.speech_translation = translation
            translated += 1
            src_preview = source[:45] + ("…" if len(source) > 45 else "")
            tr_preview = translation[:45] + ("…" if len(translation) > 45 else "")
            log(SCOPE_TRANSLATE, f"{i}/{total} — {clip.pk}: \"{src_preview}\" -> \"{tr_preview}\"")
        except Exception:
            continue
        if i % COMMIT_EVERY == 0:
            session.commit()
            log(SCOPE_TRANSLATE, f"{i}/{total} committed")

    session.commit()
    session.close()
    log(SCOPE_TRANSLATE, f"done — {translated}/{total} translated")
