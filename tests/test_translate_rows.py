"""Unit tests for core.translate.translate_rows shared helper.

The helper batches eligible rows through ``GemmaTranslator.translate_batch``
(length-bucketed for low pad waste) and falls back to per-item
``translate_text`` only when a whole batch raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _Row:
    id: int
    source: str
    source_lang: str
    translation: str | None = None


class _FakeTranslator:
    """Records batch + single calls. Responses keyed by source text;
    default echoes ``text.upper()``. ``batch_raises`` simulates a whole-batch
    failure (OOM etc.); ``single_errors`` simulates per-item failures on the
    fallback path."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        model_id: str = "fake",
        responses: dict[str, str] | None = None,
        single_errors: set[str] | None = None,
        batch_raises: bool = False,
    ) -> None:
        self.device = device
        self.model_id = model_id
        self._responses = responses or {}
        self._single_errors = single_errors or set()
        self._batch_raises = batch_raises
        self.batch_calls: list[list[tuple[str, str, str]]] = []
        self.text_calls: list[str] = []

    def _resp(self, text: str) -> str:
        return self._responses.get(text, text.upper())

    def translate_batch(
        self,
        items: list[tuple[str, str, str]],
        *,
        max_new_tokens: int,
        batch_size: int,
    ) -> list[str]:
        self.batch_calls.append(list(items))
        if self._batch_raises:
            raise RuntimeError("batch boom")
        return [self._resp(text) for (text, _src, _dst) in items]

    def translate_text(
        self,
        *,
        text: str,
        source_lang_code: str,
        target_lang_code: str,
        max_new_tokens: int,
    ) -> str:
        self.text_calls.append(text)
        if text in self._single_errors:
            raise RuntimeError("single boom")
        return self._resp(text)


class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


def _run(rows, translator, monkeypatch, **overrides):
    from core import translate as ct

    monkeypatch.setattr(ct, "GemmaTranslator", lambda model_id: translator)
    kwargs: dict[str, Any] = dict(
        get_source=lambda r: r.source,
        get_source_lang=lambda r: r.source_lang,
        set_translation=lambda r, v: setattr(r, "translation", v),
        model_id="fake",
        target_lang="en",
        max_chars=100,
        max_new_tokens=64,
        commit_every=2,
        session=_FakeSession(),
        progress_label="t",
        log_tag_prefix="row",
        seal_label="t-seal",
        batch_size=16,
    )
    kwargs.update(overrides)
    ct.translate_rows(rows, **kwargs)


def _mixed_rows() -> list[_Row]:
    return [
        _Row(id=1, source="hola", source_lang="es"),
        _Row(id=2, source="hello", source_lang="en"),  # English -> skipped
        _Row(id=3, source="bonjour", source_lang="fr"),
        _Row(id=4, source="", source_lang="ru"),  # empty -> skipped
    ]


def test_batches_eligible_rows_and_skips_english_and_empty(monkeypatch):
    translator = _FakeTranslator()
    rows = _mixed_rows()
    _run(rows, translator, monkeypatch)

    assert rows[0].translation == "HOLA"
    assert rows[1].translation is None  # English
    assert rows[2].translation == "BONJOUR"
    assert rows[3].translation is None  # empty source
    # exactly one batch holding only the two eligible rows; no single fallback
    assert len(translator.batch_calls) == 1
    assert {t for (t, _s, _d) in translator.batch_calls[0]} == {"hola", "bonjour"}
    assert translator.text_calls == []


def test_target_lang_threaded_into_batch_items(monkeypatch):
    translator = _FakeTranslator()
    rows = [_Row(id=1, source="hola", source_lang="es")]
    _run(rows, translator, monkeypatch, target_lang="de")
    assert translator.batch_calls[0][0] == ("hola", "es", "de")


def test_length_bucketing_sorts_items_ascending(monkeypatch):
    translator = _FakeTranslator()
    rows = [
        _Row(id=1, source="aaaa", source_lang="es"),
        _Row(id=2, source="a", source_lang="fr"),
        _Row(id=3, source="aaaaaa", source_lang="de"),
        _Row(id=4, source="aa", source_lang="it"),
    ]
    _run(rows, translator, monkeypatch, batch_size=16)
    texts = [t for (t, _s, _d) in translator.batch_calls[0]]
    assert texts == ["a", "aa", "aaaa", "aaaaaa"]


def test_chunks_into_batches_of_batch_size(monkeypatch):
    translator = _FakeTranslator()
    rows = [_Row(id=i, source="x" * i, source_lang="es") for i in range(1, 6)]
    _run(rows, translator, monkeypatch, batch_size=2)
    assert [len(b) for b in translator.batch_calls] == [2, 2, 1]


def test_whole_batch_failure_falls_back_to_single_items(monkeypatch):
    # batch raises -> each item retried via translate_text; the failing item is
    # isolated (left NULL) while the rest still translate.
    translator = _FakeTranslator(batch_raises=True, single_errors={"hola"})
    rows = [
        _Row(id=1, source="hola", source_lang="es"),
        _Row(id=2, source="bonjour", source_lang="fr"),
    ]
    _run(rows, translator, monkeypatch)
    assert rows[0].translation is None
    assert rows[1].translation == "BONJOUR"
    assert set(translator.text_calls) == {"hola", "bonjour"}


def test_empty_translation_left_null(monkeypatch):
    translator = _FakeTranslator(responses={"hola": ""})
    rows = [
        _Row(id=1, source="hola", source_lang="es"),
        _Row(id=2, source="bonjour", source_lang="fr"),
    ]
    _run(rows, translator, monkeypatch)
    assert rows[0].translation is None
    assert rows[1].translation == "BONJOUR"


def test_commits_during_processing(monkeypatch):
    from core import translate as ct

    translator = _FakeTranslator()
    monkeypatch.setattr(ct, "GemmaTranslator", lambda model_id: translator)
    rows = [_Row(id=i, source="x" * i, source_lang="es") for i in range(1, 6)]
    session = _FakeSession()
    ct.translate_rows(
        rows,
        get_source=lambda r: r.source,
        get_source_lang=lambda r: r.source_lang,
        set_translation=lambda r, v: setattr(r, "translation", v),
        model_id="fake",
        target_lang="en",
        max_chars=100,
        max_new_tokens=64,
        commit_every=2,
        session=session,
        progress_label="t",
        log_tag_prefix="row",
        seal_label="t-seal",
        batch_size=2,
    )
    assert session.commits >= 1


def test_no_rows_is_noop(monkeypatch):
    translator = _FakeTranslator()
    _run([], translator, monkeypatch)
    assert translator.batch_calls == []
    assert translator.text_calls == []
