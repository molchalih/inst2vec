"""Statistical dataset gating with pass A/B."""

from __future__ import annotations

import math

from sqlalchemy import func

from modules.console import log
from modules.database import Clip, User, get_session

SCOPE = "finalize_dataset"


def _count_for_user_clip_set(clip_ids: list[int], allowed: set[int]) -> int:
    return sum(1 for clip_id in clip_ids if clip_id in allowed)


def _text_ok_clip_ids(session) -> set[int]:
    return {
        row[0]
        for row in session.query(Clip.id)
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
    users: list[User],
    only_active_clips: bool,
    creator_min_clips: int,
    creator_robust_z_threshold: float,
) -> set[int]:
    outliers: set[int] = set()
    for user in users:
        clips = [
            c
            for c in user.clips
            if c.play_count is not None
            and (not only_active_clips or c.disqualified != 1)
        ]
        if len(clips) < creator_min_clips:
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
            if robust_z < creator_robust_z_threshold:
                outliers.add(clip.id)
    return outliers


def finalize_user_dataset(
    pass_name: str,
    target_clips_per_user: int,
    require_min_text_clips: bool,
    pass_a_recompute_from_scratch: bool,
    global_min_plays: int,
    global_min_plays_percentile: float,
    creator_robust_z_threshold: float,
    creator_min_clips: int,
) -> None:
    """Mark users/clips as disqualified(1) or eligible(0) with staged policy."""
    pass_name = (pass_name or "A").upper()
    if pass_name not in {"A", "B"}:
        raise ValueError("pass_name must be 'A' or 'B'")

    is_pass_a = pass_name == "A"
    use_text_gate = pass_name == "B" and require_min_text_clips

    session = get_session()
    users = session.query(User).order_by(User.id).all()
    if not users:
        session.close()
        return

    text_ok_ids: set[int] = set()
    if use_text_gate:
        text_ok_ids = _text_ok_clip_ids(session)

    disq_count = 0
    reason_counts = {"clip_count": 0, "text_count": 0}
    clip_reason_counts = {
        "user_disqualified": 0,
        "global_low_plays": 0,
        "creator_low_outlier": 0,
    }
    clip_kept = 0
    clip_disq = 0
    unresolved_users = 0

    parsed_users: list[User] = []
    for user in users:
        if user.parse_status != "success":
            # keep unresolved users neutral so debug BATCH_SIZE runs don't poison eligibility
            user.user_disqualified = None
            unresolved_users += 1
            continue
        parsed_users.append(user)

    # Pass A: pre-gate users with too few raw clips before computing stats
    pre_gate_users: list[User] = []
    pre_gated_clip_ids: set[int] = (
        set()
    )  # Track clips disqualified in pre-gate for later deduplication
    if is_pass_a:
        for user in parsed_users:
            if len(user.clips) < target_clips_per_user:
                user.user_disqualified = 1
                disq_count += 1
                reason_counts["clip_count"] += 1
                for clip in user.clips:
                    clip.disqualified = 1
                    clip_disq += 1
                    clip_reason_counts["user_disqualified"] += 1
                    pre_gated_clip_ids.add(clip.id)
            else:
                pre_gate_users.append(user)
    else:
        pre_gate_users = parsed_users  # Pass B: no pre-gate

    # pass A: apply statistical clip first then user-level cascade
    global_floor = 0
    creator_outlier_ids: set[int] = set()
    if is_pass_a:
        all_play_counts = [
            int(clip.play_count)
            for user in pre_gate_users
            for clip in user.clips
            if (
                clip.play_count is not None
                and (pass_a_recompute_from_scratch or clip.disqualified != 1)
            )
        ]
        percentile_floor = _quantile_int(all_play_counts, global_min_plays_percentile)
        global_floor = max(global_min_plays, percentile_floor)
        creator_outlier_ids = _creator_relative_low_outliers(
            pre_gate_users,
            only_active_clips=not pass_a_recompute_from_scratch,
            creator_min_clips=creator_min_clips,
            creator_robust_z_threshold=creator_robust_z_threshold,
        )

    for user in pre_gate_users if is_pass_a else parsed_users:
        if is_pass_a:
            active_clip_ids: list[int] = []
            stat_disq_by_clip: dict[int, bool] = {}
            for clip in user.clips:
                already_disq = (
                    not pass_a_recompute_from_scratch
                ) and clip.disqualified == 1
                is_global_low = bool(
                    global_floor > 0
                    and clip.play_count is not None
                    and int(clip.play_count) < global_floor
                )
                is_creator_low = clip.id in creator_outlier_ids
                stat_disq = already_disq or is_global_low or is_creator_low
                stat_disq_by_clip[clip.id] = stat_disq
                if not stat_disq:
                    active_clip_ids.append(clip.id)
        else:
            active_clip_ids = [clip.id for clip in user.clips if clip.disqualified != 1]
            stat_disq_by_clip = {clip.id: clip.disqualified == 1 for clip in user.clips}

        total_clips = len(active_clip_ids)
        text_count = _count_for_user_clip_set(active_clip_ids, text_ok_ids)

        bad_clip_count = total_clips < target_clips_per_user
        bad_text_count = use_text_gate and text_count < target_clips_per_user
        disqualified = bad_clip_count or bad_text_count
        user.user_disqualified = 1 if disqualified else 0

        if disqualified:
            disq_count += 1
            if bad_clip_count:
                reason_counts["clip_count"] += 1
            if bad_text_count:
                reason_counts["text_count"] += 1

        for clip in user.clips:
            was_pre_gated = clip.id in pre_gated_clip_ids
            should_disqualify_clip = disqualified or stat_disq_by_clip.get(
                clip.id, False
            )
            clip.disqualified = 1 if should_disqualify_clip else 0

            if should_disqualify_clip:
                # Only count this clip if it wasn't already counted in pre-gate
                if not was_pre_gated:
                    clip_disq += 1
                if disqualified and not was_pre_gated:
                    # Count user_disqualified reason even if clip was pre-gated
                    # (the user re-gate is what's disqualifying it now, even if stat-disq did too)
                    clip_reason_counts["user_disqualified"] += 1
                if is_pass_a:
                    if (
                        global_floor > 0
                        and clip.play_count is not None
                        and int(clip.play_count) < global_floor
                    ) and not was_pre_gated:
                        clip_reason_counts["global_low_plays"] += 1
                    if clip.id in creator_outlier_ids and not was_pre_gated:
                        clip_reason_counts["creator_low_outlier"] += 1
            else:
                clip_kept += 1

    session.commit()
    total_users = len(users)
    kept = total_users - disq_count
    eligible_clips = (
        session.query(func.count(Clip.id)).filter(Clip.disqualified == 0).scalar()
    )
    session.close()

    log(
        SCOPE,
        (
            f"pass {pass_name} done — kept {kept}/{total_users} users; kept {eligible_clips} clips "
            f"(clip_kept={clip_kept}, clip_disqualified={clip_disq}); "
            f"disqualified {disq_count}, unresolved {unresolved_users}; "
            f"policy [target={target_clips_per_user}, text_gate={int(require_min_text_clips)}, "
            f"recompute={int(pass_a_recompute_from_scratch)}, global_min={global_min_plays}, "
            f"global_pct={global_min_plays_percentile}, creator_rz={creator_robust_z_threshold}, "
            f"creator_min={creator_min_clips}] "
            f"reasons "
            f"[clip_count={reason_counts['clip_count']}, "
            f"text_count={reason_counts['text_count']}], "
            f"clip reasons [user={clip_reason_counts['user_disqualified']}, "
            f"global_low={clip_reason_counts['global_low_plays']} (<{global_floor}), "
            f"creator_low={clip_reason_counts['creator_low_outlier']} (rz<{creator_robust_z_threshold})]"
        ),
    )
