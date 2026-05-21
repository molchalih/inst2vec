"""MIR (Music Information Retrieval) pipeline using Essentia models."""

from core.config import Secrets, Settings
from modules.mir.pipeline import run_mir as _run_mir

__all__ = ["run_mir"]


def run_mir(settings: Settings, secrets: Secrets) -> None:
    """Per-clip MIR descriptors via MAEST + EffNet-Discogs."""
    _run_mir(settings, secrets)
