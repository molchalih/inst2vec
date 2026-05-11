"""Paper-facing summary table for the clips dataset."""

from __future__ import annotations

from statistics import mean, median

from sqlalchemy import func
from sqlalchemy.orm import Session

from modules.database import Clip

__all__ = ("clips_summary_to_markdown",)

TABLE_ROWS: tuple[tuple[str, str], ...] = (
    (r"$N$", "total_clips"),
    (r"$N_{\mathrm{kept}}$", "kept_clips"),
    (r"$N_{\mathrm{caption}}$", "with_caption_text"),
    (r"$N_{\mathrm{caption\_trans}}$", "with_caption_translation"),
    (r"$N_{\mathrm{speech}}$", "with_speech"),
    (r"$N_{\mathrm{speech\_trans}}$", "with_speech_translation"),
    (r"$N_{\mathrm{music}}$", "with_music"),
    (r"$\tilde{x}_{\mathrm{views}}$", "play_count_median"),
    (r"$\mu_\mathrm{views}$", "play_count_mean"),
    (r"$[\min-max]_{\mathrm{views}}$", "play_count_minmax"),
    # (r"$\tilde{x}_{\mathrm{likes}}$", "like_count_median"),
    # (r"$\mu_\mathrm{likes}$", "like_count_mean"),
    # (r"$[\min-max]_{\mathrm{likes}}$", "like_count_minmax")
)


KEPT_CLIP_FILTER = Clip.disqualified == 0


def _count(session: Session, *criteria) -> int:
    return int(session.query(func.count(Clip.pk)).filter(*criteria).scalar() or 0)


def _count_non_empty(session: Session, column, *criteria) -> int:
    return int(
        session.query(func.count(Clip.pk))
        .filter(column.is_not(None), func.trim(column) != "")
        .filter(*criteria)
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
    return (
        f"{median(values):,.0f}, {mean(values):,.1f}, {min(values):,}-{max(values):,}"
    )


def _numeric_values(session: Session, column, *criteria) -> list[int]:
    return [
        int(value)
        for (value,) in session.query(column)
        .filter(column.is_not(None))
        .filter(*criteria)
        .all()
    ]


def _fmt_median(values: list[int]) -> str:
    if not values:
        return "-"
    return f"{median(values):,.0f}"


def _fmt_mean(values: list[int]) -> str:
    if not values:
        return "-"
    return f"{mean(values):,.1f}"


def _fmt_min_max(values: list[int]) -> str:
    if not values:
        return "-"
    return f"{min(values):,}-{max(values):,}"


def _summary_cells(session: Session) -> dict[str, str]:
    total_clips = _count(session)
    kept_clips = _count(session, KEPT_CLIP_FILTER)

    caption_text_counts = _count_non_empty(session, Clip.caption_text, KEPT_CLIP_FILTER)
    caption_translation_counts = _count_non_empty(
        session, Clip.caption_translation, KEPT_CLIP_FILTER
    )
    speech_counts = _count(session, Clip.has_speech == 1, KEPT_CLIP_FILTER)
    speech_translation_counts = _count_non_empty(
        session, Clip.speech_translation, KEPT_CLIP_FILTER
    )
    music_counts = _count(session, Clip.has_music == 1, KEPT_CLIP_FILTER)
    play_counts = _numeric_values(session, Clip.play_count, KEPT_CLIP_FILTER)
    return {
        "total_clips": f"{total_clips:,}",
        "kept_clips": _fmt_count_share(kept_clips, total_clips),
        "with_caption_text": _fmt_count_share(caption_text_counts, kept_clips),
        "with_caption_translation": _fmt_count_share(
            caption_translation_counts, kept_clips
        ),
        "with_speech": _fmt_count_share(speech_counts, kept_clips),
        "with_speech_translation": _fmt_count_share(
            speech_translation_counts, kept_clips
        ),
        "with_music": _fmt_count_share(music_counts, kept_clips),
        "play_count_median": _fmt_median(play_counts),
        "play_count_mean": _fmt_mean(play_counts),
        "play_count_minmax": _fmt_min_max(play_counts),
        # "like_count_median": _fmt_median(like_counts),
        # "like_count_mean": _fmt_mean(like_counts),
        # "like_count_minmax": _fmt_min_max(like_counts),
    }


def clips_summary_to_markdown(eng) -> str:
    with Session(eng) as session:
        cells = _summary_cells(session)

    lines = ["| Metric | Value |", "|---|---:|"]
    for label, key in TABLE_ROWS:
        lines.append(f"| {label} | {cells[key]} |")
    return "\n".join(lines)


def get_clips_summary_cells(eng) -> dict[str, str]:
    with Session(eng) as session:
        return _summary_cells(session)
