"""Shared state constants and helpers for the MIR module."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from core.config import MirSettings
from core.database import AudioMIR
from core.pipeline import Stage

STAGE_MIR: Stage = Stage.MIR
SCOPE_MIR: str = "all"

# Positive-class index in every Essentia head's output vector
# (verified against each head's metadata.json["classes"]).
POS: int = 0

_MIR_CONFIG_FIELDS: tuple[str, ...] = (
    "binary_threshold",
    "topk_genre",
    "topk_moodtheme",
    "topk_instrument",
    "inference_sample_rate",
    "maest_checkpoint",
    "maest_output",
    "effnet_checkpoint",
    "effnet_embed_output",
)

_RESET_COLUMNS: tuple[str, ...] = (
    "is_mir_extracted",
    "mir_error",
    "approachability",
    "engagement",
    "danceability",
    "is_aggressive",
    "is_happy",
    "is_party",
    "is_relaxed",
    "is_sad",
    "is_acoustic",
    "is_electronic",
    "is_instrumental",
    "is_female_voice",
    "is_bright_timbre",
    "is_tonal",
    "genre_labels",
    "genre_scores",
    "moodtheme_labels",
    "moodtheme_scores",
    "instrument_labels",
    "instrument_scores",
    "audio_duration_s",
    "inference_time_ms",
)


def mir_config_payload(mir: MirSettings) -> str:
    """Stable JSON of the MirSettings fields that affect MIR outputs."""
    payload = {f: getattr(mir, f) for f in _MIR_CONFIG_FIELDS}
    return json.dumps(payload, sort_keys=True, default=str)


def reset_audio_mir(session: Session) -> None:
    """NULL every descriptor column on every AudioMIR row.

    Called on MIR config drift. Row identity (clip_id, created_at) is
    preserved; row-level idempotence in run_mir then re-fills the
    NULLed descriptors.
    """
    fields = {getattr(AudioMIR, c): None for c in _RESET_COLUMNS}
    session.query(AudioMIR).update(fields, synchronize_session=False)
    session.commit()
