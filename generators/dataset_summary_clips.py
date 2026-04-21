"""Paper-facing summary table for the clips dataset."""
from __future__ import annotations

from statistics import mean, median

from sqlalchemy import func
from sqlalchemy.orm import Session

from modules.database import Clip

__all__ = ("clips_summary_to_markdown",)

TABLE_ROWS: tuple[tuple[str, str], ...] = (
    ("Total clips", "total_clips"),
    ("Clips kept", "kept_clips"),
    ("Clips disqualified", "disqualified_clips"),
    ("Clips with caption text", "with_caption_text"),
    ("Clips with caption language", "with_caption_language"),
    ("Clips with caption translation", "with_caption_translation"),
    ("Clips with speech", "with_speech"),
    ("Clips with speech transcription", "with_speech_transcription"),
    ("Clips with speech language", "with_speech_language"),
    ("Clips with speech translation", "with_speech_translation"),
    ("Clips with music", "with_music"),
    ("Clips linked to music row", "with_music_id"),
    ("Play count (median, mean, min-max)", "play_count_summary"),
    ("Like count (median, mean, min-max)", "like_count_summary"),
    ("Comment count (median, mean, min-max)", "comment_count_summary"),
    ("Reshare count (median, mean, min-max)", "reshare_count_summary"),
)


def _count(session: Session, *criteria) -> int:
    return int(session.query(func.count(Clip.pk)).filter(*criteria).scalar() or 0)


def _count_non_empty(session: Session, column) -> int:
    return int(
        session.query(func.count(Clip.pk))
        .filter(column.is_not(None), func.trim(column) != "")
        .scalar()
        or 0
    )


def _fmt_count_share(count: int, total: int) -> str:
    if total <= 0:
        return "0"
    return f"{count:,} ({count / total:.1%})"


def _fmt_distribution(values: list[int]) -> str:
    if not values:
        return "-"
    return f"{median(values):,.0f}, {mean(values):,.1f}, {min(values):,}-{max(values):,}"


def _numeric_values(session: Session, column) -> list[int]:
    return [
        int(value)
        for (value,) in session.query(column)
        .filter(column.is_not(None))
        .all()
    ]


def _summary_cells(session: Session) -> dict[str, str]:
    total_clips = _count(session)
    kept_clips = _count(session, Clip.disqualified == 0)
    disqualified_clips = _count(session, Clip.disqualified == 1)

    return {
        "total_clips": f"{total_clips:,}",
        "kept_clips": _fmt_count_share(kept_clips, total_clips),
        "disqualified_clips": _fmt_count_share(disqualified_clips, total_clips),
        "with_caption_text": _fmt_count_share(
            _count_non_empty(session, Clip.caption_text), total_clips
        ),
        "with_caption_language": _fmt_count_share(
            _count_non_empty(session, Clip.caption_language), total_clips
        ),
        "with_caption_translation": _fmt_count_share(
            _count_non_empty(session, Clip.caption_translation), total_clips
        ),
        "with_speech": _fmt_count_share(_count(session, Clip.has_speech == 1), total_clips),
        "with_speech_transcription": _fmt_count_share(
            _count_non_empty(session, Clip.speech_transcription), total_clips
        ),
        "with_speech_language": _fmt_count_share(
            _count_non_empty(session, Clip.speech_language), total_clips
        ),
        "with_speech_translation": _fmt_count_share(
            _count_non_empty(session, Clip.speech_translation), total_clips
        ),
        "with_music": _fmt_count_share(_count(session, Clip.has_music == 1), total_clips),
        "with_music_id": _fmt_count_share(_count(session, Clip.music_id.is_not(None)), total_clips),
        "play_count_summary": _fmt_distribution(_numeric_values(session, Clip.play_count)),
        "like_count_summary": _fmt_distribution(_numeric_values(session, Clip.like_count)),
        "comment_count_summary": _fmt_distribution(_numeric_values(session, Clip.comment_count)),
        "reshare_count_summary": _fmt_distribution(_numeric_values(session, Clip.reshare_count)),
    }


def clips_summary_to_markdown(eng) -> str:
    with Session(eng) as session:
        cells = _summary_cells(session)

    lines = ["| Metric | Value |", "|---|---:|"]
    for label, key in TABLE_ROWS:
        lines.append(f"| {label} | {cells[key]} |")
    return "\n".join(lines)
