"""Data-acquisition stages: CSV seeding, profile fetch, download, audio extract."""

from core.config import Secrets, Settings
from modules.ingest.audio import extract_audio, extract_audio_stage
from modules.ingest.download import download_files, fetch_file
from modules.ingest.profiles import fetch_profiles
from modules.ingest.seed import load_usernames_from_csv

__all__ = [
    "download_files",
    "extract_audio",
    "extract_audio_stage",
    "fetch_file",
    "fetch_profiles",
    "load_usernames_from_csv",
    "run_audio",
    "run_download",
    "run_profiles",
    "run_seed",
]


def run_seed(settings: Settings, secrets: Secrets) -> None:
    """Seed users from the CSV."""
    load_usernames_from_csv(csv_path=settings.paths.data_csv_path)


def run_profiles(settings: Settings, secrets: Secrets) -> None:
    """Fetch Instagram profiles + clips metadata via HikerAPI."""
    fetch_profiles(hiker_api_key=secrets.hiker_api_key)


def run_download(settings: Settings, secrets: Secrets) -> None:
    """Download videos/thumbnails/profile pics for selected clips."""
    download_files(settings.download, settings.paths)


def run_audio(settings: Settings, secrets: Secrets) -> None:
    """Extract mp3 audio from downloaded videos."""
    extract_audio_stage(settings)
