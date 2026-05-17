from __future__ import annotations

import math
import statistics

from sqlalchemy.orm import Session

from modules.database import User, UserStats
from modules.filter.predicates import _preprocessed_clips


def _median_absolute_deviation(values: list[float]) -> float:
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def calculate_user_stats(session: Session) -> None:
    session.query(UserStats).delete()
    session.flush()

    for user in session.query(User).all():
        clips = _preprocessed_clips(user)
        if not clips:
            continue

        plays = [c.play_count for c in clips]
        log_plays_vals = [math.log1p(p) for p in plays]
        durations = [c.video_duration for c in clips if c.video_duration is not None]
        taken_ats = [c.taken_at for c in clips if c.taken_at is not None]

        n = len(clips)
        max_plays = max(plays)
        median_plays = statistics.median(plays)
        total_plays = sum(plays)
        ratio = max_plays / median_plays if median_plays > 0 else None
        share_top = max_plays / total_plays if total_plays > 0 else None

        oldest = min(taken_ats) if taken_ats else None
        newest = max(taken_ats) if taken_ats else None
        span_days = (
            (newest - oldest) / 86400.0
            if oldest is not None and newest is not None
            else None
        )
        per_week = (
            n / (span_days / 7.0) if span_days is not None and span_days > 0 else None
        )

        session.add(
            UserStats(
                user_id=user.id,
                n_clips=n,
                median_plays=float(median_plays),
                mean_plays=float(statistics.mean(plays)),
                max_plays=max_plays,
                min_plays=min(plays),
                mean_log_plays=float(statistics.mean(log_plays_vals)),
                median_log_plays=float(statistics.median(log_plays_vals)),
                log_plays_std=(
                    float(statistics.pstdev(log_plays_vals))
                    if len(log_plays_vals) > 1
                    else 0.0
                ),
                log_plays_mad=float(_median_absolute_deviation(log_plays_vals)),
                top_to_median_plays_ratio=ratio,
                share_of_plays_from_top_clip=share_top,
                median_video_duration=(
                    float(statistics.median(durations)) if durations else None
                ),
                mean_video_duration=(
                    float(statistics.mean(durations)) if durations else None
                ),
                oldest_clip_taken_at=oldest,
                newest_clip_taken_at=newest,
                clip_time_span_days=span_days,
                approx_clips_per_week=per_week,
            )
        )
