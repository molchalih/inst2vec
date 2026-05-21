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


def test_audio_mir_for_uses_audio_mir_dir():
    from core.config import PathsSettings

    paths = PathsSettings(
        video_dir="v",
        model_path="m",
        profile_pic_dir="p",
        thumbnail_dir="t",
        speech_audio_dir="s",
        audio_dir="a",
        audio_mir_dir="data/audio_mir",
        data_csv_path="data/data.csv",
    )
    assert str(paths.audio_mir_for(42)) == "data/audio_mir/42.wav"
    assert paths.audio_mir_dir == "data/audio_mir"


def test_audio_extraction_settings_have_mir_defaults():
    from core.config import AudioExtractionSettings

    s = AudioExtractionSettings()
    assert s.mir_codec == "pcm_s16le"
    assert s.mir_extension == "wav"
    assert s.mir_sample_rate_hz == 16_000
    assert s.mir_channels == 1
    assert s.mir_extract_timeout_s == 60
