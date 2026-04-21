"""Paper-facing summary table for the users dataset."""
from __future__ import annotations

from statistics import mean, median

from sqlalchemy import func
from sqlalchemy.orm import Session

from modules.database import User

__all__ = ("users_summary_to_markdown",)

TABLE_ROWS: tuple[tuple[str, str], ...] = (
    ("Total users", "total_users"),
    ("Parsed users", "parsed_users"),
    ("Unresolved users", "unresolved_users"),
    ("Users kept", "kept_users"),
    ("Users disqualified", "disqualified_users"),
    ("Users with full name", "with_full_name"),
    ("Users with profile picture", "with_profile_pic"),
    ("Users with HD profile picture", "with_profile_pic_hd"),
    ("Users with city", "with_city"),
    ("Following count (median, mean, min-max)", "following_count_summary"),
)


def _count(session: Session, *criteria) -> int:
    return int(session.query(func.count(User.pk)).filter(*criteria).scalar() or 0)


def _count_non_empty(session: Session, column) -> int:
    return int(
        session.query(func.count(User.pk))
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


def _summary_cells(session: Session) -> dict[str, str]:
    total_users = _count(session)
    parsed_users = _count(session, User.parse_status == "success")
    unresolved_users = _count(session, User.parse_status != "success") + _count(
        session, User.parse_status.is_(None)
    )
    kept_users = _count(session, User.user_disqualified == 0)
    disqualified_users = _count(session, User.user_disqualified == 1)
    following_counts = [
        int(value)
        for (value,) in session.query(User.following_count)
        .filter(User.following_count.is_not(None))
        .all()
    ]

    return {
        "total_users": f"{total_users:,}",
        "parsed_users": _fmt_count_share(parsed_users, total_users),
        "unresolved_users": _fmt_count_share(unresolved_users, total_users),
        "kept_users": _fmt_count_share(kept_users, total_users),
        "disqualified_users": _fmt_count_share(disqualified_users, total_users),
        "with_full_name": _fmt_count_share(_count_non_empty(session, User.full_name), total_users),
        "with_profile_pic": _fmt_count_share(
            _count_non_empty(session, User.profile_pic_url), total_users
        ),
        "with_profile_pic_hd": _fmt_count_share(
            _count_non_empty(session, User.profile_pic_url_hd), total_users
        ),
        "with_city": _fmt_count_share(_count_non_empty(session, User.city_name), total_users),
        "following_count_summary": _fmt_distribution(following_counts),
    }


def users_summary_to_markdown(eng) -> str:
    with Session(eng) as session:
        cells = _summary_cells(session)

    lines = ["| Metric | Value |", "|---|---:|"]
    for label, key in TABLE_ROWS:
        lines.append(f"| {label} | {cells[key]} |")
    return "\n".join(lines)
