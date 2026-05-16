"""Signature checks for the captions package public API."""

import inspect

from modules import captions as captions_pkg


def test_clean_captions_accepts_cfg_and_engine():
    sig = inspect.signature(captions_pkg.clean_captions)
    params = sig.parameters
    assert "cfg" in params
    assert "engine" in params
    assert params["engine"].default is None


def test_detect_caption_language_accepts_cfg_and_engine():
    sig = inspect.signature(captions_pkg.detect_caption_language)
    params = sig.parameters
    assert "cfg" in params
    assert "engine" in params


def test_translate_captions_accepts_cfg_and_engine():
    sig = inspect.signature(captions_pkg.translate_captions)
    params = sig.parameters
    assert "cfg" in params
    assert "engine" in params


def test_process_captions_accepts_cfg_and_engine():
    sig = inspect.signature(captions_pkg.process_captions)
    params = sig.parameters
    assert "cfg" in params
    assert "engine" in params
    assert params["engine"].default is None
