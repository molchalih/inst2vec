"""Unit tests for core.translate.translate_rows shared helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class _Row:
    id: int
    source: str
    source_lang: str
    translation: str | None = None


class _FakeTranslator:
    def __init__(
        self,
        *,
        device: str = "cpu",
        model_id: str = "fake",
        responses: dict[int, str | Exception] | None = None,
    ) -> None:
        self.device = device
        self.model_id = model_id
        self._responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    def translate_text(
        self,
        *,
        text: str,
        source_lang_code: str,
        target_lang_code: str,
        max_new_tokens: int,
    ) -> str:
        self.calls.append(
            {
                "text": text,
                "src": source_lang_code,
                "dst": target_lang_code,
                "max_new_tokens": max_new_tokens,
            }
        )
        resp = self._responses.get(len(self.calls) - 1, text.upper())
        if isinstance(resp, Exception):
            raise resp
        return resp


class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


def _rows() -> list[_Row]:
    return [
        _Row(id=1, source="hola", source_lang="es"),
        _Row(id=2, source="hello", source_lang="en"),  # English -> skipped
        _Row(id=3, source="bonjour", source_lang="fr"),
        _Row(id=4, source="", source_lang="ru"),  # empty -> skipped
    ]


def test_translate_rows_translates_non_english(monkeypatch: pytest.MonkeyPatch) -> None:
    from core import translate as ct

    translator = _FakeTranslator()
    monkeypatch.setattr(ct, "GemmaTranslator", lambda model_id: translator)

    session = _FakeSession()
    rows = _rows()
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
    )

    assert rows[0].translation == "HOLA"
    assert rows[1].translation is None
    assert rows[2].translation == "BONJOUR"
    assert rows[3].translation is None
    assert len(translator.calls) == 2


def test_translate_rows_translator_exception_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import translate as ct

    translator = _FakeTranslator(responses={0: RuntimeError("boom"), 1: "OK"})
    monkeypatch.setattr(ct, "GemmaTranslator", lambda model_id: translator)

    rows = [
        _Row(id=1, source="hola", source_lang="es"),
        _Row(id=2, source="bonjour", source_lang="fr"),
    ]
    ct.translate_rows(
        rows,
        get_source=lambda r: r.source,
        get_source_lang=lambda r: r.source_lang,
        set_translation=lambda r, v: setattr(r, "translation", v),
        model_id="fake",
        target_lang="en",
        max_chars=100,
        max_new_tokens=64,
        commit_every=10,
        session=_FakeSession(),
        progress_label="t",
        log_tag_prefix="row",
        seal_label="t-seal",
    )

    assert rows[0].translation is None
    assert rows[1].translation == "OK"


def test_translate_rows_empty_translation_treated_as_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import translate as ct

    translator = _FakeTranslator(responses={0: ""})
    monkeypatch.setattr(ct, "GemmaTranslator", lambda model_id: translator)

    rows = [_Row(id=1, source="hola", source_lang="es")]
    ct.translate_rows(
        rows,
        get_source=lambda r: r.source,
        get_source_lang=lambda r: r.source_lang,
        set_translation=lambda r, v: setattr(r, "translation", v),
        model_id="fake",
        target_lang="en",
        max_chars=100,
        max_new_tokens=64,
        commit_every=10,
        session=_FakeSession(),
        progress_label="t",
        log_tag_prefix="row",
        seal_label="t-seal",
    )

    assert rows[0].translation is None
