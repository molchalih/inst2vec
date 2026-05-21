"""Shared state constants and helpers for the MIR module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy.orm import Session

from core.config import MirSettings
from core.database import AudioMIR
from core.fingerprint import stable_subset_payload
from core.pipeline import Stage

_LABELS_DIR: Path = Path(__file__).resolve().parent / "labels"

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
    "maest_input",
    "maest_output",
    "maest_patch_seconds",
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


def _label_file_hashes() -> dict[str, str]:
    """SHA-256 of every labels/*.json file content, keyed by filename (sorted)."""
    out: dict[str, str] = {}
    if not _LABELS_DIR.exists():
        return out
    for path in sorted(_LABELS_DIR.glob("*.json")):
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _checkpoint_digests(mir: MirSettings) -> dict[str, str]:
    """{<pb filename>: <sha256 or 'absent'>} for every checkpoint, sorted."""
    from modules.mir.checkpoints import _manifest, _sidecar_path

    out: dict[str, str] = {}
    for _url, target in _manifest(mir):
        side = _sidecar_path(target)
        digest = "absent"
        if target.exists() and side.exists():
            try:
                data = json.loads(side.read_text())
                if isinstance(data, dict) and isinstance(data.get("sha256"), str):
                    digest = data["sha256"]
            except (json.JSONDecodeError, OSError):
                pass
        out[target.name] = digest
    return dict(sorted(out.items()))


def mir_config_payload(mir: MirSettings) -> str:
    """Stable JSON of every input that affects MIR outputs.

    Covers: MirSettings fields in ``_MIR_CONFIG_FIELDS``, SHA-256 of every
    label file under ``_LABELS_DIR``, the sidecar digest for every checkpoint
    .pb file (``"absent"`` when the sidecar has not been written yet), and the
    per-head (filename, output_op) specs from ``EFFNET_HEAD_SPECS``.
    """
    from modules.mir.models import EFFNET_HEAD_SPECS

    base = json.loads(stable_subset_payload(mir, _MIR_CONFIG_FIELDS))
    base["labels"] = _label_file_hashes()
    base["checkpoints"] = _checkpoint_digests(mir)
    base["heads"] = {
        name: list(spec) for name, spec in sorted(EFFNET_HEAD_SPECS.items())
    }
    return json.dumps(base, sort_keys=True, default=str)


def reset_audio_mir(session: Session) -> None:
    """NULL every descriptor column on every AudioMIR row.

    Called on MIR config drift. Row identity (clip_id, created_at) is
    preserved; row-level idempotence in run_mir then re-fills the
    NULLed descriptors.
    """
    fields = {getattr(AudioMIR, c): None for c in _RESET_COLUMNS}
    session.query(AudioMIR).update(fields, synchronize_session=False)
    session.commit()
