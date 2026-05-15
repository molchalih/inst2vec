"""Translate stage: GemmaTranslator → Clip.caption_translation from caption_clean."""

from __future__ import annotations

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from modules.captions.state import SCOPE_TRANSLATE
from modules.config import CaptionsSettings
from modules.console import log, progress
from modules.database import Clip, get_engine, needs_caption_translation
from modules.external.gemma_translate import GemmaTranslator


def translate_captions(cfg: CaptionsSettings, *, engine: Engine | None = None) -> None:
    """Translate non-English clean captions with missing caption_translation.

    Translation errors are logged with the clip ID, then the loop continues.
    Failed clips leave caption_translation NULL so the next run retries.
    """
    eng = engine or get_engine()
    with Session(eng) as session:
        clips = (
            session.query(Clip)
            .filter(*needs_caption_translation())
            .order_by(Clip.id)
            .all()
        )
        if not clips:
            return

        total = len(clips)
        log(SCOPE_TRANSLATE, f"{total} captions to translate")
        translator = GemmaTranslator(model_id=cfg.translate_model)
        log(SCOPE_TRANSLATE, f"loading {translator.model_id} on {translator.device}…")
        translated = 0
        failed: list[int] = []

        with progress(total, "Translating captions") as advance:
            for i, clip in enumerate(clips, 1):
                source = (clip.caption_clean or "").strip()[: cfg.translation_max_chars]
                source_lang = (clip.caption_language or "").strip().replace("_", "-")
                if (
                    not source
                    or not source_lang
                    or source_lang.lower().startswith("en")
                ):
                    advance()
                    continue

                try:
                    translation = translator.translate_text(
                        text=source,
                        source_lang_code=source_lang,
                        target_lang_code=cfg.translate_target_lang,
                        max_new_tokens=cfg.translate_max_new_tokens,
                    )
                except Exception as exc:
                    failed.append(clip.id)
                    log(
                        SCOPE_TRANSLATE,
                        f"clip {clip.id}: translation failed ({exc!r}); left retryable",
                        level="warn",
                    )
                    advance()
                    continue

                if not translation:
                    advance()
                    continue

                clip.caption_translation = translation
                translated += 1
                src_preview = source[:45] + ("…" if len(source) > 45 else "")
                tr_preview = translation[:45] + ("…" if len(translation) > 45 else "")
                advance(detail=f'{clip.id}: "{src_preview}" → "{tr_preview}"')

                if i % cfg.commit_every == 0:
                    session.commit()
        session.commit()

        parts = [f"{translated}/{total} translated"]
        if failed:
            parts.append(f"{len(failed)} failed (ids: {failed})")
        log(SCOPE_TRANSLATE, "done — " + ", ".join(parts), level="ok")
