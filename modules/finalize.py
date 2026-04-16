"""Statistical dataset gating with pass A/B."""
from __future__ import annotations

import math
import os

from sqlalchemy import func

from modules.database import Clip, User, get_session
from modules.services import log

SCOPE = "finalize_dataset"

TARGET_CLIPS_PER_USER = int(os.environ.get("FINALIZE_TARGET_CLIPS_PER_USER", 4))
REQUIRE_MIN_TEXT_CLIPS = os.environ.get("FINALIZE_REQUIRE_MIN_TEXT_CLIPS", "0") == "1"
PASS_A_RECOMPUTE_FROM_SCRATCH = os.environ.get("FINALIZE_PASS_A_RECOMPUTE_FROM_SCRATCH", "1") == "1"
GLOBAL_MIN_PLAYS = int(os.environ.get("FINALIZE_GLOBAL_MIN_PLAYS", "0"))
GLOBAL_MIN_PLAYS_PERCENTILE = float(os.environ.get("FINALIZE_GLOBAL_MIN_PLAYS_PERCENTILE", "5"))
CREATOR_ROBUST_Z_THRESHOLD = float(os.environ.get("FINALIZE_CREATOR_ROBUST_Z_THRESHOLD", "-2.5"))
CREATOR_MIN_CLIPS = int(os.environ.get("FINALIZE_CREATOR_MIN_CLIPS", "5"))


def _is_parsed_user(user: User) -> bool:
    """Mirror parse._is_parsed with a tolerant parse-signal check.

    Some legitimate profiles can have null fields (e.g., full_name/profile_pic/following_count).
    Treat user as parsed when we have any profile signal or already attached clips.
    """
    return any([
        user.full_name is not None,
        user.profile_pic_url is not None,
        user.profile_pic_url_hd is not None,
        user.following_count is not None,
        user.city_name is not None,
        bool(user.clips),
    ])


def _count_for_user_clip_set(clip_ids: list[int], allowed: set[int]) -> int:
    return sum(1 for clip_id in clip_ids if clip_id in allowed)


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


def _creator_relative_low_outliers(users: list[User], only_active_clips: bool) -> set[int]:
    outliers: set[int] = set()
    for user in users:
        clips = [c for c in user.clips if c.play_count is not None and (not only_active_clips or c.disqualified != 1)]
        if len(clips) < CREATOR_MIN_CLIPS:
            continue

        logs = [math.log1p(int(c.play_count or 0)) for c in clips]
        med = _median(logs)
        abs_dev = [abs(x - med) for x in logs]
        mad = _median(abs_dev)
        if mad <= 1e-9:
            continue

        scale = 1.4826 * mad
        for clip, value in zip(clips, logs):
            robust_z = (value - med) / scale
            if robust_z < CREATOR_ROBUST_Z_THRESHOLD:
                outliers.add(clip.pk)
    return outliers


def finalize_user_dataset(pass_name: str = "A") -> None:
    """Mark users/clips as disqualified(1) or eligible(0) with staged policy.
    """
    pass_name = (pass_name or "A").upper()
    if pass_name not in {"A", "B"}:
        raise ValueError("pass_name must be 'A' or 'B'")

    is_pass_a = pass_name == "A"
    use_text_gate = pass_name == "B" and REQUIRE_MIN_TEXT_CLIPS

    session = get_session()
    users = session.query(User).order_by(User.pk).all()
    if not users:
        session.close()
        return

    text_ok_ids: set[int] = set()
    if use_text_gate:
        text_ok_ids = _text_ok_clip_ids(session)

    disq_count = 0
    reason_counts = {"clip_count": 0, "text_count": 0}
    clip_reason_counts = {"user_disqualified": 0, "global_low_plays": 0, "creator_low_outlier": 0}
    clip_kept = 0
    clip_disq = 0
    unresolved_users = 0

    parsed_users: list[User] = []
    for user in users:
        if not _is_parsed_user(user):
            # keep unresolved users neutral so debug BATCH_SIZE runs don't poison eligibility
            user.user_disqualified = None
            unresolved_users += 1
            continue
        parsed_users.append(user)

    # pass A: apply statistical clip first then user-level cascade
    global_floor = 0
    creator_outlier_ids: set[int] = set()
    if is_pass_a:
        all_play_counts = [
            int(clip.play_count)
            for user in parsed_users
            for clip in user.clips
            if (
                clip.play_count is not None
                and (PASS_A_RECOMPUTE_FROM_SCRATCH or clip.disqualified != 1)
            )
        ]
        percentile_floor = _quantile_int(all_play_counts, GLOBAL_MIN_PLAYS_PERCENTILE)
        global_floor = max(GLOBAL_MIN_PLAYS, percentile_floor)
        creator_outlier_ids = _creator_relative_low_outliers(
            parsed_users,
            only_active_clips=not PASS_A_RECOMPUTE_FROM_SCRATCH,
        )

    for user in parsed_users:
        if is_pass_a:
            active_clip_ids: list[int] = []
            stat_disq_by_clip: dict[int, bool] = {}
            for clip in user.clips:
                already_disq = (not PASS_A_RECOMPUTE_FROM_SCRATCH) and clip.disqualified == 1
                is_global_low = bool(
                    global_floor > 0
                    and clip.play_count is not None
                    and int(clip.play_count) < global_floor
                )
                is_creator_low = clip.pk in creator_outlier_ids
                stat_disq = already_disq or is_global_low or is_creator_low
                stat_disq_by_clip[clip.pk] = stat_disq
                if not stat_disq:
                    active_clip_ids.append(clip.pk)
        else:
            active_clip_ids = [clip.pk for clip in user.clips if clip.disqualified != 1]
            stat_disq_by_clip = {clip.pk: clip.disqualified == 1 for clip in user.clips}

        total_clips = len(active_clip_ids)
        text_count = _count_for_user_clip_set(active_clip_ids, text_ok_ids)

        bad_clip_count = total_clips < TARGET_CLIPS_PER_USER
        bad_text_count = use_text_gate and text_count < TARGET_CLIPS_PER_USER
        disqualified = bad_clip_count or bad_text_count
        user.user_disqualified = 1 if disqualified else 0

        if disqualified:
            disq_count += 1
            if bad_clip_count:
                reason_counts["clip_count"] += 1
            if bad_text_count:
                reason_counts["text_count"] += 1

        for clip in user.clips:
            should_disqualify_clip = disqualified or stat_disq_by_clip.get(clip.pk, False)
            clip.disqualified = 1 if should_disqualify_clip else 0

            if should_disqualify_clip:
                clip_disq += 1
                if disqualified:
                    clip_reason_counts["user_disqualified"] += 1
                if is_pass_a:
                    if (
                        global_floor > 0
                        and clip.play_count is not None
                        and int(clip.play_count) < global_floor
                    ):
                        clip_reason_counts["global_low_plays"] += 1
                    if clip.pk in creator_outlier_ids:
                        clip_reason_counts["creator_low_outlier"] += 1
            else:
                clip_kept += 1

    session.commit()
    total_users = len(users)
    kept = total_users - disq_count
    eligible_clips = session.query(func.count(Clip.pk)).filter(Clip.disqualified == 0).scalar()
    session.close()

    log(
        SCOPE,
        (
            f"pass {pass_name} done — kept {kept}/{total_users} users; kept {eligible_clips} clips "
            f"(clip_kept={clip_kept}, clip_disqualified={clip_disq}); "
            f"disqualified {disq_count}, unresolved {unresolved_users}; "
            f"policy [target={TARGET_CLIPS_PER_USER}, text_gate={int(REQUIRE_MIN_TEXT_CLIPS)}, "
            f"recompute={int(PASS_A_RECOMPUTE_FROM_SCRATCH)}, global_min={GLOBAL_MIN_PLAYS}, "
            f"global_pct={GLOBAL_MIN_PLAYS_PERCENTILE}, creator_rz={CREATOR_ROBUST_Z_THRESHOLD}, "
            f"creator_min={CREATOR_MIN_CLIPS}] "
            f"reasons "
            f"[clip_count={reason_counts['clip_count']}, "
            f"text_count={reason_counts['text_count']}], "
            f"clip reasons [user={clip_reason_counts['user_disqualified']}, "
            f"global_low={clip_reason_counts['global_low_plays']} (<{global_floor}), "
            f"creator_low={clip_reason_counts['creator_low_outlier']} (rz<{CREATOR_ROBUST_Z_THRESHOLD})]"
        ),
    )
