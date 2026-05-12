import inspect

from modules import parse as parse_mod


def test_fetch_profiles_accepts_explicit_params():
    sig = inspect.signature(parse_mod.fetch_profiles)
    assert "hiker_api_key" in sig.parameters
