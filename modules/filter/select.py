from __future__ import annotations

import math
import random

from sqlalchemy.orm import Session

from core.config import FilterSettings
from core.database import Clip, User
from modules.filter.predicates import eligible_clips


def select_clips(session: Session, cfg: FilterSettings) -> None:
    for clip in session.query(Clip).all():
        clip.is_selected = False
    for user in session.query(User).all():
        user.is_selected = False

    for user in session.query(User).all():
        eligible = sorted(
            eligible_clips(user),
            key=lambda c: c.play_count or 0,
            reverse=True,
        )
        if not eligible:
            continue

        pool_size = max(
            cfg.selected_clips_per_user,
            math.ceil(len(eligible) * cfg.selection_pool_percent),
        )
        pool = eligible[:pool_size]

        rng = random.Random(f"{cfg.selection_random_seed}:{user.id}")
        chosen = rng.sample(pool, cfg.selected_clips_per_user)

        for clip in chosen:
            clip.is_selected = True
        user.is_selected = True
