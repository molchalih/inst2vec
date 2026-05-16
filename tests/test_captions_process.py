"""Tests for the process_captions orchestrator."""

from modules import captions as captions_pkg
from modules.captions import process_captions
from modules.config import CaptionsSettings


def _cfg():
    return CaptionsSettings(
        commit_every=2,
        translate_model="dummy",
        translate_target_lang="en",
        translation_max_chars=1000,
        translate_max_new_tokens=200,
    )


def test_process_captions_calls_each_stage_in_order(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        captions_pkg,
        "clean_captions",
        lambda cfg, *, engine=None: calls.append("clean"),
    )
    monkeypatch.setattr(
        captions_pkg,
        "detect_caption_language",
        lambda cfg, *, engine=None: calls.append("detect"),
    )
    monkeypatch.setattr(
        captions_pkg,
        "translate_captions",
        lambda cfg, *, engine=None: calls.append("translate"),
    )
    process_captions(_cfg())
    assert calls == ["clean", "detect", "translate"]


def test_process_captions_propagates_engine(monkeypatch):
    sentinel = object()
    seen: list = []
    monkeypatch.setattr(
        captions_pkg,
        "clean_captions",
        lambda cfg, *, engine=None: seen.append(("clean", engine)),
    )
    monkeypatch.setattr(
        captions_pkg,
        "detect_caption_language",
        lambda cfg, *, engine=None: seen.append(("detect", engine)),
    )
    monkeypatch.setattr(
        captions_pkg,
        "translate_captions",
        lambda cfg, *, engine=None: seen.append(("translate", engine)),
    )
    process_captions(_cfg(), engine=sentinel)
    assert all(eng is sentinel for _, eng in seen)
