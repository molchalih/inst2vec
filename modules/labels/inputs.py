"""Stage-1 clip-input adapters per case.

Each adapter takes ``(clip, mir_row, visual_payload)`` and returns the
textual input the generator will see (or ``None`` to signal the runner to
mark ``status="failed"`` with the case-specific error string). The
adapters are thin wrappers around ``modules.embeddings.text`` helpers so
there is a single source of truth for caption / speech / MIR
verbalization across the labels and embeddings stages.
"""

from __future__ import annotations

import json

from modules.embeddings.text import (
    build_audio_text,
    build_sandwich_text,
    verbalize_mir,
)


def video_input(_clip, _mir_row, _visual_payload) -> str | None:
    """Sentinel adapter for the visual case.

    The clip-pass runner routes the video case to
    ``LabelsGenerator.run(video_path, prompt)`` directly (see
    ``LabelCaseSpec.clip_uses_video``) and never calls this function.
    Returns ``None`` so it cannot accidentally feed text into the video
    branch.
    """
    return None


def audio_input(clip, mir_row, _visual_payload) -> str | None:
    """Audio case input — speech + MIR verbalization joined by ``" | "``."""
    return build_audio_text(clip, mir_row)


def sandwich_input(clip, mir_row, visual_payload: dict | None) -> str | None:
    """Sandwich case input — visual observations + caption / speech / music.

    Returns ``None`` when the visual ``ClipLabel`` payload is absent so
    the clip-pass runner can mark the row ``status="failed",
    error="missing_video_label"`` and bump attempts. When the visual
    payload is present, the textual block is still emitted even if
    ``build_sandwich_text`` returns ``None`` — the visual payload alone
    carries enough signal to label the clip.
    """
    if visual_payload is None:
        return None
    sandwich_text = build_sandwich_text(clip, mir_row)
    visual_block = json.dumps(visual_payload, sort_keys=True)
    text_block = sandwich_text or ""
    return f"VISUAL_OBSERVATIONS={visual_block}\n\nTEXT={text_block}"


def maest_input(_clip, mir_row, _visual_payload) -> str | None:
    """Maest (music) case input — verbalize_mir output when music is detected."""
    if mir_row is None or not getattr(mir_row, "is_music_detected", False):
        return None
    return verbalize_mir(mir_row)
