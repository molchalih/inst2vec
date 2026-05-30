"""Filter pipeline: hard preprocess → user stats → soft preprocess → selection.

`process_dataset` is fingerprint-gated: on data or config drift it resets
all derived clip/user fields and re-runs the four passes; on a match it
no-ops.
"""

from __future__ import annotations

import json
import time
from itertools import chain

from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.config import FilterSettings, Secrets, Settings
from core.database import Clip, User, get_engine
from core.log import StageResult, event, scope, stage
from modules.filter.predicates import (
    eligible_clips,
    has_any_flag,
    preprocessed_clips,
    selected_clips,
)
from modules.filter.preprocess import (
    _hard_preprocess,
    _soft_preprocess,
    compute_creator_robust_stats,
    derive_eligibility,
    derive_preprocessing_status,
    derive_user_eligibility,
    flag_basic_policy_clips,
    flag_garbage_clips,
    flag_global_percentile_clips,
    flag_low_median_creators,
    flag_users_without_enough_eligible_clips,
    flag_users_without_enough_preprocessed_clips,
)
from modules.filter.select import select_clips
from modules.filter.state import (
    CLIP_EXCLUSION_FLAGS,
    HARD_CLIP_EXCLUSION_FLAGS,
    SCOPE,
    SOFT_CLIP_EXCLUSION_FLAGS,
    STAGE,
    USER_EXCLUSION_FLAGS,
    reset_dataset_processing_state,
)
from modules.filter.stats import calculate_user_stats

__all__ = [
    "CLIP_EXCLUSION_FLAGS",
    "HARD_CLIP_EXCLUSION_FLAGS",
    "SOFT_CLIP_EXCLUSION_FLAGS",
    "USER_EXCLUSION_FLAGS",
    "calculate_user_stats",
    "compute_creator_robust_stats",
    "derive_eligibility",
    "derive_preprocessing_status",
    "derive_user_eligibility",
    "eligible_clips",
    "flag_basic_policy_clips",
    "flag_garbage_clips",
    "flag_global_percentile_clips",
    "flag_low_median_creators",
    "flag_users_without_enough_eligible_clips",
    "flag_users_without_enough_preprocessed_clips",
    "has_any_flag",
    "preprocessed_clips",
    "process_dataset",
    "run",
    "select_clips",
    "selected_clips",
]


def _random_sample(session: Session, cfg: FilterSettings) -> None:
    select_clips(session, cfg)


def _fingerprint(session: Session, cfg: FilterSettings) -> fp.Fingerprint:
    user_rows = session.query(User.id).order_by(User.id).all()
    clip_rows = (
        session.query(
            Clip.id,
            Clip.user_id,
            Clip.play_count,
            Clip.video_duration,
            Clip.taken_at,
            Clip.video_url,
            Clip.like_count,
        )
        .order_by(Clip.id)
        .all()
    )
    data = fp.hash_rows(
        chain(
            (("u", *r) for r in user_rows),
            (("c", *r) for r in clip_rows),
        )
    )
    config = fp.hash_text(json.dumps(cfg.model_dump(), sort_keys=True, default=str))
    dependency = fp.hash_text("")
    return fp.Fingerprint(data=data, config=config, dependency=dependency)


@scope("filter")
def process_dataset(
    cfg: FilterSettings,
    *,
    engine=None,
) -> None:
    eng = engine or get_engine()
    with Session(eng) as session:
        current = _fingerprint(session, cfg)
        if not fp.is_stale(session, STAGE, SCOPE, current):
            event("SKIP", "fingerprint")
            return

        diff = fp.describe_diff(session, STAGE, SCOPE, current)
        event("SCAN", "fingerprint", stats={"diff": diff})

        reset_dataset_processing_state(session)

        t_hard = time.perf_counter()
        _hard_preprocess(session, cfg)
        event("WRITE", "hard", stats={"time": time.perf_counter() - t_hard})

        calculate_user_stats(session)

        t_soft = time.perf_counter()
        _soft_preprocess(session, cfg)
        event("WRITE", "soft", stats={"time": time.perf_counter() - t_soft})

        t_sel = time.perf_counter()
        _random_sample(session, cfg)
        event("WRITE", "selection", stats={"time": time.perf_counter() - t_sel})

        fp.mark_complete(session, STAGE, SCOPE, current)
        session.commit()


@stage("filter")
def run(settings: Settings, secrets: Secrets) -> StageResult:
    """Filter dataset to eligible+selected clips."""
    process_dataset(settings.filter)
    return StageResult()
