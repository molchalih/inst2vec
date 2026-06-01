"""TranslateGemma stage: translate non-English transcriptions."""

from __future__ import annotations

from core.database import Clip, clip_needs_speech_translation, get_session
from core.log import scope
from core.translate import RowAccessors, translate_rows


@scope("speech")
def translate_speech(
    commit_every: int,
    translate_model: str,
    translate_target_lang: str,
    translation_max_chars: int,
    translate_max_new_tokens: int,
    translate_batch_size: int = 16,
) -> None:
    """Translate clips with detected non-English speech but no translation.

    Failed translations (empty result or exception) leave
    ``speech_translation`` NULL so the next run can retry.
    """
    session = get_session()
    try:
        clips = (
            session.query(Clip)
            .filter(*clip_needs_speech_translation())
            .order_by(Clip.id)
            .all()
        )
        translate_rows(
            clips,
            accessors=RowAccessors(
                get_source=lambda c: c.speech_transcription,
                get_source_lang=lambda c: c.speech_language,
                set_translation=lambda c, v: setattr(c, "speech_translation", v),
            ),
            model_id=translate_model,
            target_lang=translate_target_lang,
            max_chars=translation_max_chars,
            max_new_tokens=translate_max_new_tokens,
            commit_every=commit_every,
            batch_size=translate_batch_size,
            session=session,
            progress_label="Translating speech",
            log_tag_prefix="clip",
            seal_label="speech-translate",
        )
        session.commit()
    finally:
        session.close()
