import inspect

from modules import music as music_mod


def test_classify_music_accepts_params():
    sig = inspect.signature(music_mod.classify_music)
    for name in ("music", "paths", "secrets"):
        assert name in sig.parameters, f"missing: {name}"


def test_extract_music_features_accepts_params():
    sig = inspect.signature(music_mod.extract_music_features)
    for name in ("music", "paths", "secrets"):
        assert name in sig.parameters, f"missing: {name}"
