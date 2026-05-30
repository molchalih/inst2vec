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

from core.console import log as _log
from core.console import progress
from core.lang import is_english
from core.log import _scope_var as _log_scope_var
from core.log import event
from core.vendor.gemma_translate import GemmaTranslator


def _translate_singly(
    translator: GemmaTranslator,
    chunk: list[tuple[Any, str, str]],
    target_lang: str,
    max_new_tokens: int,
    log_tag_prefix: str,
    mt_scope: str,
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
            _log(
                mt_scope,
                "EXTRACT",
                f"{log_tag_prefix}_{row.id}",
                "ERR",
                stats={"err": repr(exc)},
            )
            results.append(None)
    return results


def _collect_eligible(
    rows: list[Any],
    get_source: Callable[[Any], str | None],
    get_source_lang: Callable[[Any], str | None],
    max_chars: int,
) -> tuple[list[tuple[Any, str, str]], int]:
    """Keep non-empty, non-English sources; length-bucket them ascending.

    Returns ``(eligible, skipped)``. Length-bucketing means each batch holds
    similar-length sources: with greedy decoding the batch runs until its
    longest member finishes, so mixing a 5-char source with a 1000-char one
    wastes compute on padding.
    """
    eligible: list[tuple[Any, str, str]] = []
    skipped = 0
    for row in rows:
        source = (get_source(row) or "").strip()[:max_chars]
        source_lang = (get_source_lang(row) or "").strip().replace("_", "-")
        if not source or not source_lang or is_english(source_lang):
            skipped += 1
            continue
        eligible.append((row, source, source_lang))
    eligible.sort(key=lambda rs: len(rs[1]))
    return eligible, skipped


def _translate_chunk(
    translator: GemmaTranslator,
    chunk: list[tuple[Any, str, str]],
    target_lang: str,
    max_new_tokens: int,
    width: int,
    log_tag_prefix: str,
    mt_scope: str,
) -> list[str | None]:
    """Translate one chunk, degrading a whole-batch failure to per-row."""
    items = [(src, lang, target_lang) for (_row, src, lang) in chunk]
    try:
        return list(
            translator.translate_batch(
                items, max_new_tokens=max_new_tokens, batch_size=width
            )
        )
    except Exception:
        # Whole-batch failure (e.g. CUDA OOM): isolate per row.
        return _translate_singly(
            translator, chunk, target_lang, max_new_tokens, log_tag_prefix, mt_scope
        )


def _store_chunk(
    chunk: list[tuple[Any, str, str]],
    results: list[str | None],
    set_translation: Callable[[Any, str], None],
    target_lang: str,
    log_tag_prefix: str,
    mt_scope: str,
    advance: Callable[..., None],
) -> int:
    """Persist successful translations from one chunk. Returns the success count."""
    batch_ok = 0
    for (row, source, source_lang), translation in zip(chunk, results, strict=True):
        tag = f"{log_tag_prefix}_{row.id}"
        if translation is None:  # already logged ERR in the fallback
            advance()
            continue
        if not translation:
            _log(mt_scope, "EXTRACT", tag, "WARN", stats={"src": source_lang})
            advance()
            continue
        set_translation(row, translation)
        batch_ok += 1
        _log(
            mt_scope,
            "EXTRACT",
            tag,
            "ok",
            stats={"src": source_lang, "dst": target_lang},
        )
        src_preview = source[:45] + ("…" if len(source) > 45 else "")
        tr_preview = translation[:45] + ("…" if len(translation) > 45 else "")
        advance(detail=f'{row.id}: "{src_preview}" → "{tr_preview}"')
    return batch_ok


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
    log_scope: str = "",
) -> None:
    """Translate ``rows`` whose source is non-empty and non-English.

    Caller is responsible for the query that produced ``rows`` and for the
    final ``session.commit()``. This helper commits roughly every
    ``commit_every`` rows for crash-resume, then logs a SEAL line.

    ``batch_size`` is the GPU decode batch width; it affects only throughput,
    not the per-row outputs, so it is intentionally excluded from the speech /
    captions config fingerprints.

    ``log_scope`` is accepted for backward compatibility but unused — log lines
    pick up the active scope from the caller's ContextVar.
    """
    del log_scope  # unused; scope comes from caller's ContextVar
    total = len(rows)
    if total == 0:
        return

    event("SCAN", progress_label, stats={"todo": total})
    t_load = time.perf_counter()
    translator = GemmaTranslator(model_id=model_id)
    event(
        "LOAD",
        translator.model_id,
        stats={
            "time": time.perf_counter() - t_load,
            "device": str(translator.device),
        },
    )

    # Pre-filter: only non-empty, non-English sources reach the GPU. Skipped
    # rows are accounted for in the progress bar up front.
    eligible, skipped = _collect_eligible(rows, get_source, get_source_lang, max_chars)

    width = max(batch_size, 1)
    translated = 0
    since_commit = 0
    t_stage = time.perf_counter()
    # Capture the active log scope once; translate.py goes through the raw
    # _log escape hatch as it runs outside a @scope/@stage context.
    _mt_scope = _log_scope_var.get() or "translate"

    with progress(total, progress_label) as advance:
        if skipped:
            advance(skipped)
        for start in range(0, len(eligible), width):
            chunk = eligible[start : start + width]
            t0 = time.perf_counter()
            results = _translate_chunk(
                translator,
                chunk,
                target_lang,
                max_new_tokens,
                width,
                log_tag_prefix,
                _mt_scope,
            )
            batch_ok = _store_chunk(
                chunk,
                results,
                set_translation,
                target_lang,
                log_tag_prefix,
                _mt_scope,
                advance,
            )
            translated += batch_ok

            _log(
                _mt_scope,
                "EXTRACT",
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

    event(
        "SEAL",
        seal_label,
        stats={
            "translated": translated,
            "of": total,
            "time": time.perf_counter() - t_stage,
        },
    )
