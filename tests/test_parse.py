import inspect

from modules import parse as parse_mod


def test_fetch_profiles_accepts_explicit_params():
    sig = inspect.signature(parse_mod.fetch_profiles)
    for name in ("batch_size", "max_clips", "hiker_api_key"):
        assert name in sig.parameters, f"missing param: {name}"
