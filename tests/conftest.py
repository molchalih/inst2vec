import os
import shutil
import subprocess

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("IDENTITY_DB_URL", "sqlite:///:memory:")


@pytest.fixture(scope="session", autouse=True)
def _init_test_db():
    """Initialize main + identity DB for the whole test session."""
    from core.database import init_db

    init_db("sqlite:///:memory:", "sqlite:///:memory:")


@pytest.fixture(scope="session")
def sample_mp4_with_audio(tmp_path_factory):
    """Tiny synthetic mp4 (5s, h264 + 44.1kHz stereo aac).

    Skipped if ffmpeg is not on PATH.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x64:rate=10:duration=5",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=5",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-shortest",
        str(out),
    ]
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return out
