"""Speech pipeline: VAD pre-gate + classify (Whisper) + translate (TranslateGemma)."""

from modules.speech.classify import classify_speech, clean_speech
from modules.speech.translate import translate_speech
from modules.speech.vad import VadConfig, VadResult, prepare_for_whisper

__all__ = [
    "VadConfig",
    "VadResult",
    "classify_speech",
    "clean_speech",
    "prepare_for_whisper",
    "translate_speech",
]
