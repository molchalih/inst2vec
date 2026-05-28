"""Per-case orchestrator for the cluster-labelling pass.

Mirrors ``modules/labels/pipeline.py`` for the clip pass, but operates
per ``embedding_case`` and per cluster_id. Pure orchestration: the
sampling, prompt building, validation and storage helpers are all in
sibling modules.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.config import LabelsSettings
from core.database import Clip, ClipLabel, ClusterLabel, StageState, UserCluster
from core.log import event, item, scope
from core.pipeline import Stage
from modules.labels.cluster_render import (
    ClipCandidate,
    estimate_tokens,
    pick_clips,
    render_prompt_body,
)
from modules.labels.prompts import prompt_for_cluster
from modules.labels.state import (
    STAGE_CLUSTER_LABELS,
    cluster_labels_config_payload,
    cluster_scope_for,
)
from modules.labels.store import bump_failure, upsert_success, upsert_terminal_failure
from modules.labels.validation import validate_cluster


class _ClusterMember:
    __slots__ = ("centrality", "clips", "user_id")

    def __init__(self, user_id: int, centrality: float) -> None:
        self.user_id = user_id
        self.centrality = centrality
        self.clips: list[ClipCandidate] = []


def _fingerprint_for(
    session: Session,
    *,
    case: str,
    labels: LabelsSettings,
    candidates_per_cluster: dict[int, list[ClipCandidate]],
) -> fp.Fingerprint:
    rows: list[tuple] = []
    for cid in sorted(candidates_per_cluster):
        for cand in candidates_per_cluster[cid]:
            rows.append((cid, cand.clip_id))
    return fp.Fingerprint(
        data=fp.hash_rows(rows),
        config=fp.hash_text(cluster_labels_config_payload(labels, case=case)),
        dependency=fp.compose_hashes(
            fp.stage_dependency_hash(session, Stage.LABELS, case),
            fp.stage_dependency_hash(session, Stage.CLUSTER_ASSIGN, case),
        ),
    )


def _load_candidates(
    session: Session,
    *,
    case: str,
    labels: LabelsSettings,
) -> dict[int, list[ClipCandidate]]:
    members_by_cluster: dict[int, list[_ClusterMember]] = defaultdict(list)
    uc_rows = (
        session.execute(select(UserCluster).where(UserCluster.embedding_case == case))
        .scalars()
        .all()
    )
    if not uc_rows:
        return {}
    member_by_user: dict[int, _ClusterMember] = {}
    for uc in uc_rows:
        if uc.cluster_id < 0:
            continue
        m = _ClusterMember(user_id=uc.user_id, centrality=float(uc.centrality or 0.0))
        member_by_user[uc.user_id] = m
        members_by_cluster[int(uc.cluster_id)].append(m)
    if not member_by_user:
        return {}

    user_ids = list(member_by_user)
    label_rows = session.execute(
        select(Clip, ClipLabel)
        .join(ClipLabel, ClipLabel.clip_id == Clip.id)
        .where(Clip.user_id.in_(user_ids))
        .where(Clip.is_selected.is_(True))
        .where(ClipLabel.label_case == case)
        .where(ClipLabel.status == "success")
    ).all()
    for clip, label in label_rows:
        m = member_by_user.get(clip.user_id)
        if m is None:
            continue
        m.clips.append(
            ClipCandidate(
                clip_id=clip.id,
                warning_count=len(label.warnings or []),
                payload=label.payload or {},
            )
        )

    prompt_overhead = estimate_tokens(prompt_for_cluster(labels, case=case))
    out: dict[int, list[ClipCandidate]] = {}
    for cid, members in members_by_cluster.items():
        picked = pick_clips(
            members,
            prompt_overhead_tokens=prompt_overhead,
            max_per_user=labels.cluster_max_clips_per_user,
            max_clips_total=labels.cluster_max_clips_per_cluster,
            token_budget=labels.cluster_sample_token_budget,
        )
        out[cid] = picked
    return out


def _attempt_one(
    session: Session,
    *,
    case: str,
    cluster_id: int,
    candidates: list[ClipCandidate],
    labels: LabelsSettings,
    generator,
) -> None:
    key = (case, cluster_id)
    if not candidates:
        upsert_terminal_failure(
            session,
            ClusterLabel,
            key=key,
            error="no_input",
            attempts=labels.cluster_max_attempts,
        )
        return

    prompt = (
        prompt_for_cluster(labels, case=case) + "\n\n" + render_prompt_body(candidates)
    )
    try:
        raw = generator.run_text(prompt, max_new_tokens=labels.cluster_max_new_tokens)
    except Exception as exc:
        bump_failure(
            session,
            ClusterLabel,
            key=key,
            error=f"runtime:{exc}",
            max_attempts=labels.cluster_max_attempts,
        )
        return
    payload, status, warnings = validate_cluster(raw, labels, case=case)
    if status == "failed":
        bump_failure(
            session,
            ClusterLabel,
            key=key,
            error=warnings[0] if warnings else "validation",
            max_attempts=labels.cluster_max_attempts,
        )
        return
    assert payload is not None
    upsert_success(
        session,
        ClusterLabel,
        key=key,
        validation=status,
        payload=payload,
        warnings=warnings,
    )
    row = session.get(ClusterLabel, key)
    assert row is not None
    row.sampled_clip_ids = [c.clip_id for c in candidates]


def _pending_cluster_ids(
    session: Session,
    *,
    case: str,
    all_ids: Sequence[int],
    max_attempts: int,
) -> list[int]:
    rows = session.execute(
        select(
            ClusterLabel.cluster_id, ClusterLabel.status, ClusterLabel.attempts
        ).where(ClusterLabel.embedding_case == case)
    ).all()
    state = {cid: (status, attempts or 0) for cid, status, attempts in rows}
    out: list[int] = []
    for cid in all_ids:
        s = state.get(cid)
        if s is None:
            out.append(cid)
            continue
        status, attempts = s
        if status == "pending" or (status == "failed" and attempts < max_attempts):
            out.append(cid)
    return out


def _run_case(
    *,
    session: Session,
    case: str,
    labels: LabelsSettings,
    generator,
) -> None:
    candidates_per_cluster = _load_candidates(session, case=case, labels=labels)
    current = _fingerprint_for(
        session,
        case=case,
        labels=labels,
        candidates_per_cluster=candidates_per_cluster,
    )

    def _wipe(s: Session) -> None:
        s.execute(delete(ClusterLabel).where(ClusterLabel.embedding_case == case))

    # Missing stage_state means either "never ran" (cluster_labels empty,
    # wipe is a no-op) or "crashed before mark_complete / restored without
    # state" (cluster_labels may carry rows produced under an unknown
    # prior config). Treat it as drift up front so the gate's "no prior
    # state" branch can't seal stale payloads under the current fingerprint.
    if session.get(StageState, (STAGE_CLUSTER_LABELS, cluster_scope_for(case))) is None:
        _wipe(session)
        session.commit()

    fp.gate(
        session,
        STAGE_CLUSTER_LABELS,
        cluster_scope_for(case),
        current,
        on_drift=_wipe,
        check_dependency=True,
        check_data=True,
    )
    session.commit()

    all_ids = sorted(candidates_per_cluster)
    pending = _pending_cluster_ids(
        session,
        case=case,
        all_ids=all_ids,
        max_attempts=labels.cluster_max_attempts,
    )
    if not pending:
        fp.mark_complete(
            session, STAGE_CLUSTER_LABELS, cluster_scope_for(case), current
        )
        session.commit()
        return

    event(
        "GET",
        "qwen3-vl-cluster",
        stats={"case": case, "clusters": len(pending)},
    )
    # One attempt per cluster per pipeline run; failures retried on
    # subsequent runs via ``_pending_cluster_ids``. Each attempt commits
    # so a crash mid-pass is recoverable.
    for cid in pending:
        candidates = candidates_per_cluster.get(cid, [])
        with item("WRITE", f"{case}/{cid}"):
            _attempt_one(
                session,
                case=case,
                cluster_id=cid,
                candidates=candidates,
                labels=labels,
                generator=generator,
            )
            session.commit()

    fp.mark_complete(session, STAGE_CLUSTER_LABELS, cluster_scope_for(case), current)
    session.commit()


def run_all_cases(
    *,
    session: Session,
    labels: LabelsSettings,
    generator,
    cases: Sequence[str],
) -> None:
    """Per-case loop; reuses the already-loaded generator across cases."""
    for case in cases:
        _run_case_scoped(session=session, case=case, labels=labels, generator=generator)


@scope("labels:clusters:{case}")
def _run_case_scoped(
    *,
    session: Session,
    case: str,
    labels: LabelsSettings,
    generator,
) -> None:
    _run_case(session=session, case=case, labels=labels, generator=generator)
