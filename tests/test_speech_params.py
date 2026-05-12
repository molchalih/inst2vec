import inspect

from modules import speech as speech_mod


def test_classify_speech_accepts_params():
    sig = inspect.signature(speech_mod.classify_speech)
    for name in (
        "video_dir",
        "whisper_model",
        "commit_every",
        "logprob_threshold",
        "compression_threshold",
        "min_meaningful_chars",
    ):
        assert name in sig.parameters, f"missing: {name}"


def test_translate_speech_accepts_params():
    sig = inspect.signature(speech_mod.translate_speech)
    for name in (
        "commit_every",
        "translate_model",
        "translate_target_lang",
        "translation_max_chars",
        "translate_max_new_tokens",
    ):
        assert name in sig.parameters, f"missing: {name}"
