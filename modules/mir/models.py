"""In-process registry for MIR model graphs.

Constructs MAEST + EffNet once per ``run_mir`` invocation. Per-head
output ops are pinned here based on each model's verified
``metadata.json`` (see ``schema.outputs[*].name``):

  * Regression heads (approachability, engagement) -> ``model/Identity``
  * Binary classification heads (danceability, mood_*, voice_instrumental,
    gender, timbre, tonal_atonal) -> ``model/Softmax``
  * Multi-tag classification heads (mtg_jamendo_moodtheme,
    mtg_jamendo_instrument) -> ``model/Sigmoid``
"""

from __future__ import annotations

from pathlib import Path

from core.config import MirSettings
from core.vendor.effnet import EffNet
from core.vendor.maest import MAEST

_SOFTMAX = "model/Softmax"
_SIGMOID = "model/Sigmoid"
_IDENTITY = "model/Identity"

# (filename, output_op) for every EffNet head exposed via run_mir.
EFFNET_HEAD_SPECS: dict[str, tuple[str, str]] = {
    "approachability": ("approachability_regression-discogs-effnet-1.pb", _IDENTITY),
    "engagement": ("engagement_regression-discogs-effnet-1.pb", _IDENTITY),
    "danceability": ("danceability-discogs-effnet-1.pb", _SOFTMAX),
    "mood_aggressive": ("mood_aggressive-discogs-effnet-1.pb", _SOFTMAX),
    "mood_happy": ("mood_happy-discogs-effnet-1.pb", _SOFTMAX),
    "mood_party": ("mood_party-discogs-effnet-1.pb", _SOFTMAX),
    "mood_relaxed": ("mood_relaxed-discogs-effnet-1.pb", _SOFTMAX),
    "mood_sad": ("mood_sad-discogs-effnet-1.pb", _SOFTMAX),
    "mood_acoustic": ("mood_acoustic-discogs-effnet-1.pb", _SOFTMAX),
    "mood_electronic": ("mood_electronic-discogs-effnet-1.pb", _SOFTMAX),
    "voice_instrumental": ("voice_instrumental-discogs-effnet-1.pb", _SOFTMAX),
    "gender": ("gender-discogs-effnet-1.pb", _SOFTMAX),
    "timbre": ("timbre-discogs-effnet-1.pb", _SOFTMAX),
    "tonal_atonal": ("tonal_atonal-discogs-effnet-1.pb", _SOFTMAX),
    "moodtheme": ("mtg_jamendo_moodtheme-discogs-effnet-1.pb", _SIGMOID),
    "instrument": ("mtg_jamendo_instrument-discogs-effnet-1.pb", _SIGMOID),
}


def build_maest(mir: MirSettings) -> MAEST:
    """Construct a MAEST graph from the configured checkpoint."""
    return MAEST(
        pb=Path(mir.model_dir) / mir.maest_checkpoint,
        output=mir.maest_output,
    )


def build_effnet(mir: MirSettings) -> EffNet:
    """Construct an EffNet + per-head graph bundle."""
    root = Path(mir.model_dir)
    heads = {
        name: (root / filename, output_op)
        for name, (filename, output_op) in EFFNET_HEAD_SPECS.items()
    }
    return EffNet(
        embed_pb=root / mir.effnet_checkpoint,
        heads=heads,
        embed_output=mir.effnet_embed_output,
    )
