"""TranslateGemma stage: translate non-English transcriptions."""

from __future__ import annotations

from core.console import log, progress
from core.database import Clip, clip_needs_speech_translation, get_session
from core.vendor.gemma_translate import GemmaTranslator
from modules.speech.state import SCOPE_TRANSLATE


def translate_speech(
    commit_every: int,
    translate_model: str,
    translate_target_lang: str,
    translation_max_chars: int,
    translate_max_new_tokens: int,
) -> None:
    """Translate all clips that have detected non-English speech but no translation.

    Failed translations (empty result or exception) leave ``speech_translation``
    NULL so the next run can retry.
    """
    session = get_session()
    clips = (
        session.query(Clip)
        .filter(*clip_needs_speech_translation())
        .order_by(Clip.id)
        .all()
    )
    if not clips:
        session.close()
        return

    total = len(clips)
    log(SCOPE_TRANSLATE, f"{total} clips to translate")
    translator = GemmaTranslator(model_id=translate_model)
    log(SCOPE_TRANSLATE, f"loading {translator.model_id} on {translator.device}…")
    translated = 0

    with progress(total, "Translating speech") as advance:
        for i, clip in enumerate(clips, 1):
            source = (clip.speech_transcription or "").strip()[:translation_max_chars]
            source_lang = (clip.speech_language or "").strip().replace("_", "-")
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
            except Exception:
                advance()
                continue
            if not translation:
                advance()
                continue
            clip.speech_translation = translation
            translated += 1
            src_preview = source[:45] + ("…" if len(source) > 45 else "")
            tr_preview = translation[:45] + ("…" if len(translation) > 45 else "")
            advance(detail=f'{clip.id}: "{src_preview}" → "{tr_preview}"')

            if i % commit_every == 0:
                session.commit()

    session.commit()
    session.close()
    log(SCOPE_TRANSLATE, f"done — {translated}/{total} translated", level="ok")
