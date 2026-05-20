"""Shared Gemma-driven row-translation loop.

Used by ``modules/speech/translate.py`` and ``modules/captions/translate.py``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from core.console import log, progress
from core.lang import is_english
from core.vendor.gemma_translate import GemmaTranslator


def translate_rows(
    rows: list[Any],
    *,
    get_source: Callable[[Any], str | None],
    get_source_lang: Callable[[Any], str | None],
    set_translation: Callable[[Any, str], None],
    model_id: str,
    target_lang: str,
    max_chars: int,
    max_new_tokens: int,
    commit_every: int,
    session: Any,
    progress_label: str,
    log_tag_prefix: str,
    seal_label: str,
    log_scope: str = "gemma",
) -> None:
    """Translate ``rows`` whose source is non-empty and non-English.

    Caller is responsible for the query that produced ``rows`` and for the
    final ``session.commit()``. This helper commits every ``commit_every``
    rows for crash-resume, then logs a SEAL line.
    """
    total = len(rows)
    if total == 0:
        return

    log(log_scope, "SCAN", progress_label, "ok", stats={"todo": total})
    t_load = time.perf_counter()
    translator = GemmaTranslator(model_id=model_id)
    log(
        log_scope,
        "LOAD",
        translator.model_id,
        "ok",
        stats={
            "time": time.perf_counter() - t_load,
            "device": str(translator.device),
        },
    )

    translated = 0
    t_stage = time.perf_counter()

    with progress(total, progress_label) as advance:
        for i, row in enumerate(rows, 1):
            source = (get_source(row) or "").strip()[:max_chars]
            source_lang = (get_source_lang(row) or "").strip().replace("_", "-")
            if not source or not source_lang or is_english(source_lang):
                advance()
                continue

            t0 = time.perf_counter()
            tag = f"{log_tag_prefix}_{row.id}"
            try:
                translation = translator.translate_text(
                    text=source,
                    source_lang_code=source_lang,
                    target_lang_code=target_lang,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:
                log(
                    log_scope,
                    "MT",
                    tag,
                    "ERR",
                    stats={
                        "time": time.perf_counter() - t0,
                        "err": f"translator: {exc!r}",
                    },
                )
                advance()
                continue

            if not translation:
                log(
                    log_scope,
                    "MT",
                    tag,
                    "none",
                    stats={"time": time.perf_counter() - t0, "src": source_lang},
                )
                advance()
                continue

            set_translation(row, translation)
            translated += 1
            log(
                log_scope,
                "MT",
                tag,
                "ok",
                stats={
                    "time": time.perf_counter() - t0,
                    "src": source_lang,
                    "dst": target_lang,
                },
            )
            src_preview = source[:45] + ("…" if len(source) > 45 else "")
            tr_preview = translation[:45] + ("…" if len(translation) > 45 else "")
            advance(detail=f'{row.id}: "{src_preview}" → "{tr_preview}"')

            if i % commit_every == 0:
                session.commit()

    log(
        log_scope,
        "SEAL",
        seal_label,
        "ok",
        stats={
            "translated": translated,
            "of": total,
            "time": time.perf_counter() - t_stage,
        },
    )
