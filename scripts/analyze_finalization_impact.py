#!/usr/bin/env python
"""Preview how much the dataset shrinks under finalization policy."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from sqlalchemy import func

from modules.database import Clip, User, get_session

load_dotenv()

TARGET_CLIPS_PER_USER = int(os.environ.get("FINALIZE_TARGET_CLIPS_PER_USER", 4))
REQUIRE_MIN_TEXT_CLIPS = os.environ.get("FINALIZE_REQUIRE_MIN_TEXT_CLIPS", "0") == "1"
PASS_A_RECOMPUTE_FROM_SCRATCH = (
    os.environ.get("FINALIZE_PASS_A_RECOMPUTE_FROM_SCRATCH", "1") == "1"
)
GLOBAL_MIN_PLAYS = int(os.environ.get("FINALIZE_GLOBAL_MIN_PLAYS", "0"))
GLOBAL_MIN_PLAYS_PERCENTILE = float(
    os.environ.get("FINALIZE_GLOBAL_MIN_PLAYS_PERCENTILE", "5")
)
CREATOR_ROBUST_Z_THRESHOLD = float(
    os.environ.get("FINALIZE_CREATOR_ROBUST_Z_THRESHOLD", "-2.5")
)
CREATOR_MIN_CLIPS = int(os.environ.get("FINALIZE_CREATOR_MIN_CLIPS", "5"))


def _pct(n: int, total: int) -> str:
    return f"{(100 * n / total):.1f}%" if total else "N/A"


def _policy_reasons(total_clips: int, text_count: int) -> list[str]:
    reasons: list[str] = []
    bad_clip_count = total_clips < TARGET_CLIPS_PER_USER
    bad_text_count = REQUIRE_MIN_TEXT_CLIPS and text_count < TARGET_CLIPS_PER_USER

    if bad_clip_count:
        reasons.append("clip_count")
    if bad_text_count:
        reasons.append("text_count")
    return reasons


def _text_ok_clip_ids(session) -> set[int]:
    return {
        row[0]
        for row in session.query(Clip.pk)
        .filter(
            ((Clip.caption_text.is_not(None)) & (func.trim(Clip.caption_text) != ""))
            | (
                (Clip.has_speech == 1)
                & (Clip.speech_transcription.is_not(None))
                & (func.trim(Clip.speech_transcription) != "")
            )
        )
        .all()
    }


def _quantile_int(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)
    ordered = sorted(values)
    idx = min(int((percentile / 100.0) * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[idx]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _creator_relative_low_outliers(
    users: list[User], only_active_clips: bool
) -> set[int]:
    outliers: set[int] = set()
    for user in users:
        clips = [
            c
            for c in user.clips
            if c.play_count is not None
            and (not only_active_clips or c.disqualified != 1)
        ]
        if len(clips) < CREATOR_MIN_CLIPS:
            continue
        logs = [math.log1p(int(c.play_count or 0)) for c in clips]
        med = _median(logs)
        abs_dev = [abs(x - med) for x in logs]
        mad = _median(abs_dev)
        if mad <= 1e-9:
            continue
        scale = 1.4826 * mad
        for clip, value in zip(clips, logs, strict=False):
            robust_z = (value - med) / scale
            if robust_z < CREATOR_ROBUST_Z_THRESHOLD:
                outliers.add(clip.pk)
    return outliers


def main() -> None:
    session = get_session()
    try:
        users = session.query(User).order_by(User.pk).all()
        if not users:
            print("No users found.")
            return

        parsed_users = [u for u in users if u.parse_status == "success"]
        unresolved_users = len(users) - len(parsed_users)

        text_ok_ids: set[int] = set()
        if REQUIRE_MIN_TEXT_CLIPS:
            text_ok_ids = _text_ok_clip_ids(session)

        total_users = len(users)
        total_clips = session.query(func.count(Clip.pk)).scalar() or 0

        projected_disq_users = 0
        projected_kept_users = unresolved_users
        projected_disq_clips = projected_kept_clips = 0
        reason_counts = {"clip_count": 0, "text_count": 0}
        clip_reason_counts = {
            "user_disqualified": 0,
            "global_low_plays": 0,
            "creator_low_outlier": 0,
        }
        sample_disq = []

        play_vals = [
            int(clip.play_count)
            for user in parsed_users
            for clip in user.clips
            if (
                clip.play_count is not None
                and (PASS_A_RECOMPUTE_FROM_SCRATCH or clip.disqualified != 1)
            )
        ]
        percentile_floor = _quantile_int(play_vals, GLOBAL_MIN_PLAYS_PERCENTILE)
        global_floor = max(GLOBAL_MIN_PLAYS, percentile_floor)
        creator_outlier_ids = _creator_relative_low_outliers(
            parsed_users,
            only_active_clips=not PASS_A_RECOMPUTE_FROM_SCRATCH,
        )

        # Unresolved users are neutral in finalize pass A/B (user_disqualified stays NULL),
        # so preview should carry their current clip flags through unchanged.
        for user in users:
            if user.parse_status == "success":
                continue
            for clip in user.clips:
                if clip.disqualified == 1:
                    projected_disq_clips += 1
                else:
                    projected_kept_clips += 1

        for user in parsed_users:
            active_clip_ids: list[int] = []
            for clip in user.clips:
                already_disq = (
                    not PASS_A_RECOMPUTE_FROM_SCRATCH
                ) and clip.disqualified == 1
                is_global_low = bool(
                    global_floor > 0
                    and clip.play_count is not None
                    and int(clip.play_count) < global_floor
                )
                is_creator_low = clip.pk in creator_outlier_ids
                stat_disq = already_disq or is_global_low or is_creator_low
                if not stat_disq:
                    active_clip_ids.append(clip.pk)

            total_user_clips = len(active_clip_ids)
            text_count = sum(1 for clip_id in active_clip_ids if clip_id in text_ok_ids)
            reasons = _policy_reasons(total_user_clips, text_count)
            user_disq = bool(reasons)

            if user_disq:
                projected_disq_users += 1
                for r in reasons:
                    reason_counts[r] += 1
                if len(sample_disq) < 15:
                    sample_disq.append(
                        (
                            user.username,
                            total_user_clips,
                            text_count,
                            ",".join(reasons),
                        )
                    )
            else:
                projected_kept_users += 1

            for clip in user.clips:
                already_disq = clip.disqualified == 1
                if PASS_A_RECOMPUTE_FROM_SCRATCH:
                    already_disq = False
                is_global_low = bool(
                    global_floor > 0
                    and clip.play_count is not None
                    and int(clip.play_count) < global_floor
                )
                is_creator_low = clip.pk in creator_outlier_ids
                stat_disq = already_disq or is_global_low or is_creator_low
                if user_disq or stat_disq:
                    projected_disq_clips += 1
                    if user_disq:
                        clip_reason_counts["user_disqualified"] += 1
                    if is_global_low:
                        clip_reason_counts["global_low_plays"] += 1
                    if is_creator_low:
                        clip_reason_counts["creator_low_outlier"] += 1
                else:
                    projected_kept_clips += 1

        print("=" * 62)
        print("inst2vec — Finalization Impact Preview (read-only)")
        print("=" * 62)
        print("\nPolicy from .env:")
        print(f"  FINALIZE_TARGET_CLIPS_PER_USER={TARGET_CLIPS_PER_USER}")
        print(f"  FINALIZE_REQUIRE_MIN_TEXT_CLIPS={int(REQUIRE_MIN_TEXT_CLIPS)}")
        print(
            f"  FINALIZE_PASS_A_RECOMPUTE_FROM_SCRATCH={int(PASS_A_RECOMPUTE_FROM_SCRATCH)}"
        )
        print(f"  FINALIZE_GLOBAL_MIN_PLAYS={GLOBAL_MIN_PLAYS}")
        print(f"  FINALIZE_GLOBAL_MIN_PLAYS_PERCENTILE={GLOBAL_MIN_PLAYS_PERCENTILE}")
        print(f"  FINALIZE_CREATOR_ROBUST_Z_THRESHOLD={CREATOR_ROBUST_Z_THRESHOLD}")
        print(f"  FINALIZE_CREATOR_MIN_CLIPS={CREATOR_MIN_CLIPS}")
        print(f"  derived_global_floor={global_floor}")
        print(f"  unresolved_users={unresolved_users}")

        print("\nProjected result if disqualified users are removed:")
        print(
            f"  users kept:         {projected_kept_users:>6,} / {total_users:,} ({_pct(projected_kept_users, total_users)})"
        )
        print(
            f"  users removed:      {projected_disq_users:>6,} / {total_users:,} ({_pct(projected_disq_users, total_users)})"
        )
        print(
            f"  clips kept:         {projected_kept_clips:>6,} / {total_clips:,} ({_pct(projected_kept_clips, total_clips)})"
        )
        print(
            f"  clips removed:      {projected_disq_clips:>6,} / {total_clips:,} ({_pct(projected_disq_clips, total_clips)})"
        )

        print("\nDisqualification reason counts (users can have multiple):")
        print(f"  clip_count:         {reason_counts['clip_count']:>6,}")
        print(f"  text_count:         {reason_counts['text_count']:>6,}")
        print("\nClip disqualification reason counts (clips can have multiple):")
        print(f"  user_disqualified:  {clip_reason_counts['user_disqualified']:>6,}")
        print(f"  global_low_plays:   {clip_reason_counts['global_low_plays']:>6,}")
        print(f"  creator_low_outlier:{clip_reason_counts['creator_low_outlier']:>6,}")

        if sample_disq:
            print("\nSample disqualified users:")
            print("  username                    clips  text   reasons")
            for username, clips, texts, reasons in sample_disq:
                print(f"  {username[:25]:<25} {clips:>5}  {texts:>5}  {reasons}")
        print()
    finally:
        session.close()


if __name__ == "__main__":
    main()
