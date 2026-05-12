import inspect

from modules import download as dl_mod


def test_download_files_accepts_explicit_params():
    sig = inspect.signature(dl_mod.download_files)
    for name in (
        "batch_size",
        "max_clips",
        "max_attempts",
        "retry_delay",
        "profile_pic_dir",
        "thumbnail_dir",
        "video_dir",
    ):
        assert name in sig.parameters, f"missing param: {name}"
