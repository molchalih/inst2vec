import inspect

from modules import music as music_mod


def test_classify_music_accepts_params():
    sig = inspect.signature(music_mod.classify_music)
    for name in ("music", "paths", "secrets"):
        assert name in sig.parameters, f"missing: {name}"


def test_extract_music_features_accepts_params():
    sig = inspect.signature(music_mod.extract_music_features)
    for name in (
        "video_dir",
        "http_timeout",
        "commit_every",
        "spotify_client_id",
        "spotify_client_secret",
        "spotify_token_skew_seconds",
        "spotify_search_limit",
        "spotify_request_timeout",
        "reccobeats_batch_size",
        "reccobeats_delay_min",
        "reccobeats_delay_max",
        "manual_features_max_seconds",
        "manual_features_sample_rate",
        "manual_features_max_mb",
        "manual_features_mp3_bitrate",
    ):
        assert name in sig.parameters, f"missing: {name}"
