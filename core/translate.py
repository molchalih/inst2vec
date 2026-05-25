"""Shared Gemma-driven row-translation loop.

Used by ``modules/speech/translate.py`` and ``modules/captions/translate.py``.

Eligible rows are translated in length-bucketed batches through the GPU
(``GemmaTranslator.translate_batch``) so the decoder is not run one sequence at
a time. A whole batch that raises (e.g. CUDA OOM) degrades to per-item
``translate_text`` so a single bad row can't void the rest of the batch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from core.console import log, progress
from core.lang import is_english
from core.vendor.gemma_translate import GemmaTranslator


def _translate_singly(
    translator: GemmaTranslator,
    chunk: list[tuple[Any, str, str]],
    target_lang: str,
    max_new_tokens: int,
    log_scope: str,
    log_tag_prefix: str,
) -> list[str | None]:
    """Fallback for a failed batch: translate each row on its own.

    Returns a list aligned to ``chunk``; an item that raises yields ``None``
    (already logged as ERR here) so the caller leaves it NULL for retry, while
    successful items yield their (possibly empty) translation string.
    """
    results: list[str | None] = []
    for row, source, source_lang in chunk:
        try:
            results.append(
                translator.translate_text(
                    text=source,
                    source_lang_code=source_lang,
                    target_lang_code=target_lang,
                    max_new_tokens=max_new_tokens,
                )
            )
        except Exception as exc:
            log(
                log_scope,
                "MT",
                f"{log_tag_prefix}_{row.id}",
                "ERR",
                stats={"err": f"translator: {exc!r}"},
            )
            results.append(None)
    return results


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
    batch_size: int = 16,
    log_scope: str = "gemma",
) -> None:
    """Translate ``rows`` whose source is non-empty and non-English.

    Caller is responsible for the query that produced ``rows`` and for the
    final ``session.commit()``. This helper commits roughly every
    ``commit_every`` rows for crash-resume, then logs a SEAL line.

    ``batch_size`` is the GPU decode batch width; it affects only throughput,
    not the per-row outputs, so it is intentionally excluded from the speech /
    captions config fingerprints.
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

    # Pre-filter: only non-empty, non-English sources reach the GPU. Skipped
    # rows are accounted for in the progress bar up front.
    eligible: list[tuple[Any, str, str]] = []
    skipped = 0
    for row in rows:
        source = (get_source(row) or "").strip()[:max_chars]
        source_lang = (get_source_lang(row) or "").strip().replace("_", "-")
        if not source or not source_lang or is_english(source_lang):
            skipped += 1
            continue
        eligible.append((row, source, source_lang))

    # Length-bucket so each batch holds similar-length sources: with greedy
    # decoding the batch runs until its longest member finishes, so mixing a
    # 5-char source with a 1000-char one wastes compute on padding.
    eligible.sort(key=lambda rs: len(rs[1]))

    width = max(batch_size, 1)
    translated = 0
    since_commit = 0
    t_stage = time.perf_counter()

    with progress(total, progress_label) as advance:
        if skipped:
            advance(skipped)
        for start in range(0, len(eligible), width):
            chunk = eligible[start : start + width]
            items = [(src, lang, target_lang) for (_row, src, lang) in chunk]

            t0 = time.perf_counter()
            try:
                results: list[str | None] = list(
                    translator.translate_batch(
                        items, max_new_tokens=max_new_tokens, batch_size=width
                    )
                )
            except Exception:
                # Whole-batch failure (e.g. CUDA OOM): isolate per row.
                results = _translate_singly(
                    translator,
                    chunk,
                    target_lang,
                    max_new_tokens,
                    log_scope,
                    log_tag_prefix,
                )

            batch_ok = 0
            for (row, source, source_lang), translation in zip(
                chunk, results, strict=True
            ):
                tag = f"{log_tag_prefix}_{row.id}"
                if translation is None:  # already logged ERR in the fallback
                    advance()
                    continue
                if not translation:
                    log(log_scope, "MT", tag, "none", stats={"src": source_lang})
                    advance()
                    continue
                set_translation(row, translation)
                translated += 1
                batch_ok += 1
                log(
                    log_scope,
                    "MT",
                    tag,
                    "ok",
                    stats={"src": source_lang, "dst": target_lang},
                )
                src_preview = source[:45] + ("…" if len(source) > 45 else "")
                tr_preview = translation[:45] + ("…" if len(translation) > 45 else "")
                advance(detail=f'{row.id}: "{src_preview}" → "{tr_preview}"')

            log(
                log_scope,
                "MT",
                "batch",
                "ok",
                stats={
                    "n": len(chunk),
                    "ok": batch_ok,
                    "time": time.perf_counter() - t0,
                },
            )

            since_commit += len(chunk)
            if since_commit >= commit_every:
                session.commit()
                since_commit = 0

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
