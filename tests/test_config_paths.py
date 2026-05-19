from pathlib import Path

from core.config import PathsSettings


def _paths() -> PathsSettings:
    return PathsSettings(
        video_dir="data/videos",
        model_path="/m",
        profile_pic_dir="data/pp",
        thumbnail_dir="data/thumbs",
        speech_audio_dir="data/speech",
        audio_dir="data/audio",
        data_csv_path="data/data.csv",
    )


def test_video_for():
    assert _paths().video_for(42) == Path("data/videos") / "42.mp4"


def test_audio_for():
    assert _paths().audio_for(42) == Path("data/audio") / "42.mp3"


def test_thumbnail_for():
    assert _paths().thumbnail_for(42) == Path("data/thumbs") / "42.jpg"
