"""Speech pipeline: classify (Whisper) + translate (TranslateGemma)."""

from modules.speech.classify import classify_speech, clean_speech
from modules.speech.translate import translate_speech

__all__ = ["classify_speech", "clean_speech", "translate_speech"]
