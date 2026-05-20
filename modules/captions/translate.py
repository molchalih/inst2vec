"""Translate stage: GemmaTranslator → Clip.caption_translation from caption_clean."""

from __future__ import annotations

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from core.config import CaptionsSettings
from core.database import Clip, get_engine, needs_caption_translation
from core.translate import translate_rows


def translate_captions(cfg: CaptionsSettings, *, engine: Engine | None = None) -> None:
    """Translate non-English clean captions with missing caption_translation.

    Translation errors leave ``caption_translation`` NULL so the next run
    retries.
    """
    eng = engine or get_engine()
    with Session(eng) as session:
        clips = (
            session.query(Clip)
            .filter(*needs_caption_translation())
            .order_by(Clip.id)
            .all()
        )
        translate_rows(
            clips,
            get_source=lambda c: c.caption_clean,
            get_source_lang=lambda c: c.caption_language,
            set_translation=lambda c, v: setattr(c, "caption_translation", v),
            model_id=cfg.translate_model,
            target_lang=cfg.translate_target_lang,
            max_chars=cfg.translation_max_chars,
            max_new_tokens=cfg.translate_max_new_tokens,
            commit_every=cfg.commit_every,
            session=session,
            progress_label="Translating captions",
            log_tag_prefix="cap",
            seal_label="captions-translate",
        )
        session.commit()
