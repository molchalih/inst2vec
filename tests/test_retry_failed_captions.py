"""Test: modules/captions/retry.py::retry_failed_captions delegates to process_captions."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database import Base, Clip, User


@pytest.fixture(autouse=True)
def _ensure_repo_on_path():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def test_retry_calls_process_captions(monkeypatch):
    from modules.captions import retry as captions_retry

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1, parse_status="success"))
        s.add(
            Clip(
                id=1,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                caption_text="hola",
            )
        )
        s.commit()

    monkeypatch.setattr(captions_retry, "get_engine", lambda: eng)

    called = {}

    def fake_process(cfg, *, engine=None):
        called["cfg"] = cfg
        called["engine"] = engine

    monkeypatch.setattr(captions_retry, "process_captions", fake_process)

    cfg = SimpleNamespace(
        commit_every=2,
        translate_model="dummy",
        translate_target_lang="en",
        translation_max_chars=1000,
        translate_max_new_tokens=200,
    )
    captions_retry.retry_failed_captions(cfg)

    assert called["cfg"] is cfg
    assert called["engine"] is eng
