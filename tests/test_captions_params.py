import inspect

from modules import captions as captions_mod


def test_clean_captions_accepts_commit_every():
    sig = inspect.signature(captions_mod.clean_captions)
    assert "commit_every" in sig.parameters


def test_translate_captions_accepts_params():
    sig = inspect.signature(captions_mod.translate_captions)
    for name in (
        "commit_every",
        "translate_model",
        "translate_target_lang",
        "translation_max_chars",
        "translate_max_new_tokens",
    ):
        assert name in sig.parameters, f"missing: {name}"
