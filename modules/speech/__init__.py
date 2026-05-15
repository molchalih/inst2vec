"""Speech pipeline: classify (Whisper) + translate (TranslateGemma).

Transitional shim: re-exports the legacy flat-module public API until
classify.py / translate.py exist (Tasks 6 + 7 + 8).
"""

from importlib import import_module as _import_module

_legacy = _import_module("modules._speech_flat")
classify_speech = _legacy.classify_speech
clean_speech = _legacy.clean_speech
translate_speech = _legacy.translate_speech

__all__ = ["classify_speech", "clean_speech", "translate_speech"]
