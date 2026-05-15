"""Paper-facing summary table for the users dataset."""

from __future__ import annotations

from statistics import mean, median

from sqlalchemy import func, inspect
from sqlalchemy.orm import Session

from modules.database import Clip, User
from modules.eligibility import is_eligible


def _table_exists(bind, table_name: str) -> bool:
    return table_name in inspect(bind).get_table_names()


def _table_has_columns(bind, table_name: str, *column_names: str) -> bool:
    columns = {c["name"] for c in inspect(bind).get_columns(table_name)}
    return all(name in columns for name in column_names)


__all__ = ("users_summary_to_markdown",)


TABLE_ROWS: tuple[tuple[str, str], ...] = (
    (r"$N$", "total_users"),
    (r"$N_{\mathrm{kept}}$", "kept_users"),
    (r"$\tilde{x}_{\mathrm{following}}$", "following_count_median"),
    (r"$\mu_\mathrm{following}$", "following_count_mean"),
    (r"$[\min-max]_{\mathrm{following}}$", "following_count_minmax"),
    (r"$\tilde{x}_{\mathrm{views}}$", "play_count_per_user_median"),
    (r"$\mu_\mathrm{views}$", "play_count_per_user_mean"),
)


def _count(session: Session, *criteria) -> int:
    return int(session.query(func.count(User.id)).filter(*criteria).scalar() or 0)


def _fmt_count_share(count: int, total: int) -> str:
    if total <= 0:
        return "0"
    return f"{count:,} ({count / total:.1%})"


def _fmt_distribution(values: list[float]) -> tuple[str, str, str]:
    if not values:
        return "-", "-", "-"
    return (
        f"{median(values):,.0f}",
        f"{mean(values):,.1f}",
        f"{min(values):,.0f}-{max(values):,.0f}",
    )


def _kept_play_count_distribution(session: Session) -> list[float]:
    if not _table_exists(session.get_bind(), "clips") or not _table_has_columns(
        session.get_bind(), "clips", "play_count", "eligibility"
    ):
        return []

    rows = (
        session.query(Clip.user_id, func.avg(Clip.play_count).label("avg_play_count"))
        .join(User, Clip.user_id == User.id)
        .filter(
            User.is_eligible.is_(True),
            is_eligible(Clip.eligibility),
            Clip.play_count.is_not(None),
        )
        .group_by(Clip.user_id)
        .all()
    )
    return [
        float(avg_play_count)
        for _, avg_play_count in rows
        if avg_play_count is not None
    ]


def _summary_cells(session: Session) -> dict[str, str]:
    total_users = _count(session)
    kept_users = _count(session, User.is_eligible.is_(True))
    kept_user_filters = (User.is_eligible.is_(True),)

    following_counts = [
        int(value)
        for (value,) in session.query(User.following_count)
        .filter(User.following_count.is_not(None), *kept_user_filters)
        .all()
    ]
    following_distribution = _fmt_distribution([float(v) for v in following_counts])

    user_play_averages = _kept_play_count_distribution(session)
    play_distribution = _fmt_distribution(user_play_averages)

    return {
        "total_users": f"{total_users:,}",
        "kept_users": _fmt_count_share(kept_users, total_users),
        "following_count_median": following_distribution[0],
        "following_count_mean": following_distribution[1],
        "following_count_minmax": following_distribution[2],
        "play_count_per_user_median": play_distribution[0],
        "play_count_per_user_mean": play_distribution[1],
        "play_count_per_user_minmax": play_distribution[2],
    }


def users_summary_to_markdown(eng) -> str:
    with Session(eng) as session:
        cells = _summary_cells(session)

    lines = ["| Metric | Value |", "|---|---:|"]
    for label, key in TABLE_ROWS:
        lines.append(f"| {label} | {cells[key]} |")
    return "\n".join(lines)


def get_users_summary_cells(eng) -> dict[str, str]:
    with Session(eng) as session:
        return _summary_cells(session)
