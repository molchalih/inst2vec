import inspect

from modules import _embeddings_legacy as emb_mod


def test_embed_video_clips_accepts_params():
    sig = inspect.signature(emb_mod.embed_video_clips)
    for name in (
        "model_path",
        "video_dir",
        "embed_max_length",
        "adaptive_max_frames",
        "adaptive_default_fps",
        "exclude_disqualified_users",
    ):
        assert name in sig.parameters, f"missing: {name}"


def test_embed_sandwich_clips_accepts_params():
    sig = inspect.signature(emb_mod.embed_sandwich_clips)
    for name in (
        "model_path",
        "video_dir",
        "embed_max_length",
        "adaptive_max_frames",
        "adaptive_default_fps",
        "exclude_disqualified_users",
    ):
        assert name in sig.parameters, f"missing: {name}"


def test_embed_audio_clips_accepts_params():
    sig = inspect.signature(emb_mod.embed_audio_clips)
    for name in (
        "model_path",
        "video_dir",
        "embed_max_length",
        "adaptive_max_frames",
        "adaptive_default_fps",
        "exclude_disqualified_users",
    ):
        assert name in sig.parameters, f"missing: {name}"
