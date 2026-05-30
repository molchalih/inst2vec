"""Per-case orchestrator for the cluster-labelling pass.

Mirrors ``modules/labels/pipeline.py`` for the clip pass, but operates
per ``embedding_case`` and per cluster_id. Pure orchestration: the
sampling, prompt building, validation and storage helpers are all in
sibling modules.

Two evidence-loading paths share the same downstream sampling / prompt
/ validation / store machinery, dispatched by ``spec.runs_clip_pass``:

* stage-1-backed cases (currently only ``video``) — join
  ``Clip ⋈ ClipLabel`` for the case and use the validated
  ``ClipLabel.payload`` dict as each member clip's evidence;
* stage-1-skipped cases (sandwich, auditory, spoken, textual) — call
  ``spec.clip_input(clip, mir_row, visual_payload)`` per member clip and
  use the returned raw-evidence string directly. The cluster prompt for
  these cases synthesises straight from caption / speech / MIR text
  (plus the upstream video ``ClipLabel`` JSON for sandwich).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.config import LabelsSettings
from core.database import (
    AudioMIR,
    Clip,
    ClipLabel,
    ClusterLabel,
    StageState,
    UserCluster,
    clip_used_in_analysis,
)
from core.log import event, item, scope
from core.pipeline import Stage
from modules.labels.cases import REGISTRY, LabelCaseSpec
from modules.labels.cluster_render import (
    ClipCandidate,
    estimate_tokens,
    pick_clips,
    render_prompt_body,
)
from modules.labels.prompts import prompt_for_cluster
from modules.labels.schema import cluster_schema
from modules.labels.state import (
    STAGE_CLUSTER_LABELS,
    cluster_labels_config_payload,
    cluster_scope_for,
)
from modules.labels.store import bump_failure, upsert_success, upsert_terminal_failure
from modules.labels.validation import (
    _CLUSTER_LABEL_MAX_CHARS,
    format_failure_error,
    validate_cluster,
)


class _ClusterMember:
    __slots__ = ("centrality", "clips", "user_id")

    def __init__(self, user_id: int, centrality: float) -> None:
        self.user_id = user_id
        self.centrality = centrality
        self.clips: list[ClipCandidate] = []


def _cluster_dependency_hash(session: Session, *, spec: LabelCaseSpec) -> str:
    """Compose the cluster fingerprint's dependency slot for ``spec``.

    Stage-1-backed cases inherit upstream drift through their own ``LABELS``
    stage state (sealed by ``clip_pass.run_case``). Stage-1-skipped cases
    hash the raw upstream stages directly (captions / speech / MIR) plus
    any ``consumes_label_cases`` ``LABELS`` rows (the video case for
    sandwich). ``CLUSTER_ASSIGN`` is always included.
    """
    parts: list[str] = []
    if spec.runs_clip_pass:
        parts.append(fp.stage_dependency_hash(session, Stage.LABELS, spec.name))
    else:
        for st, sc in spec.stage1_dependency_stages:
            parts.append(fp.stage_dependency_hash(session, st, sc))
        for dep_case in spec.consumes_label_cases:
            parts.append(fp.stage_dependency_hash(session, Stage.LABELS, dep_case))
    parts.append(fp.stage_dependency_hash(session, Stage.CLUSTER_ASSIGN, spec.name))
    return fp.compose_hashes(*parts)


def _candidate_payload_hash(payload: dict | str) -> str:
    """Deterministic content hash of a ``ClipCandidate.payload``.

    Stage-1-skipped cases must drift the cluster fingerprint when their
    per-clip evidence text changes — captions/speech/MIR all seal
    config-only ``StageState`` fingerprints (their ``data`` slot is
    ``hash_text("")``), so ``stage_dependency_hash`` does NOT notice a
    transcript or descriptor change. Folding the payload hash into the
    cluster fingerprint's ``data`` slot is the only place per-clip
    content drift can land. ``dict`` payloads are sorted-key JSON
    encoded; ``str`` payloads are hashed verbatim.
    """
    if isinstance(payload, str):
        return fp.hash_text(payload)
    return fp.hash_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    )


def _fingerprint_for(
    session: Session,
    *,
    spec: LabelCaseSpec,
    labels: LabelsSettings,
    candidates_per_cluster: dict[int, list[ClipCandidate]],
) -> fp.Fingerprint:
    rows: list[tuple] = []
    for cid in sorted(candidates_per_cluster):
        for cand in candidates_per_cluster[cid]:
            rows.append((cid, cand.clip_id, _candidate_payload_hash(cand.payload)))
    return fp.Fingerprint(
        data=fp.hash_rows(rows),
        config=fp.hash_text(cluster_labels_config_payload(labels, case=spec.name)),
        dependency=_cluster_dependency_hash(session, spec=spec),
    )


def _collect_members(
    session: Session, *, case: str
) -> tuple[dict[int, list[_ClusterMember]], dict[int, _ClusterMember]]:
    """Build per-cluster member lists and a user → member index for ``case``."""
    members_by_cluster: dict[int, list[_ClusterMember]] = defaultdict(list)
    member_by_user: dict[int, _ClusterMember] = {}
    uc_rows = (
        session.execute(select(UserCluster).where(UserCluster.embedding_case == case))
        .scalars()
        .all()
    )
    for uc in uc_rows:
        if uc.cluster_id < 0:
            continue
        m = _ClusterMember(user_id=uc.user_id, centrality=float(uc.centrality or 0.0))
        member_by_user[uc.user_id] = m
        members_by_cluster[int(uc.cluster_id)].append(m)
    return members_by_cluster, member_by_user


def _populate_from_stage1(
    session: Session,
    *,
    case: str,
    member_by_user: dict[int, _ClusterMember],
) -> None:
    """Attach validated ``ClipLabel.payload`` blobs to each member's clip list."""
    label_rows = session.execute(
        select(Clip, ClipLabel)
        .join(ClipLabel, ClipLabel.clip_id == Clip.id)
        .where(Clip.user_id.in_(list(member_by_user)))
        .where(*clip_used_in_analysis())
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


def _populate_from_raw_signals(
    session: Session,
    *,
    spec: LabelCaseSpec,
    member_by_user: dict[int, _ClusterMember],
) -> None:
    """Build raw per-clip evidence strings via ``spec.clip_input``.

    Clips whose adapter returns ``None`` (e.g. missing video ClipLabel for
    sandwich, no music detected for auditory) are silently dropped —
    the same clip is dropped from every cluster the user belongs to, and
    a cluster left with zero usable clips later fails with ``no_input``
    in ``_prepare_request``.
    """
    clip_rows = (
        session.execute(
            select(Clip).where(
                Clip.user_id.in_(list(member_by_user)),
                *clip_used_in_analysis(),
            )
        )
        .scalars()
        .all()
    )
    if not clip_rows:
        return
    clip_ids = [c.id for c in clip_rows]
    mir_by_clip: dict[int, AudioMIR] = {
        m.clip_id: m
        for m in (
            session.execute(select(AudioMIR).where(AudioMIR.clip_id.in_(clip_ids)))
            .scalars()
            .all()
        )
    }
    visual_by_clip: dict[int, dict] = {}
    if spec.consumes_label_cases:
        assert len(spec.consumes_label_cases) == 1, (
            f"{spec.name} consumes_label_cases must be 0 or 1"
        )
        dep_case = spec.consumes_label_cases[0]
        rows = (
            session.execute(
                select(ClipLabel).where(
                    ClipLabel.clip_id.in_(clip_ids),
                    ClipLabel.label_case == dep_case,
                    ClipLabel.status == "success",
                )
            )
            .scalars()
            .all()
        )
        visual_by_clip = {r.clip_id: r.payload or {} for r in rows}

    for clip in clip_rows:
        m = member_by_user.get(clip.user_id)
        if m is None:
            continue
        visual_payload = (
            visual_by_clip.get(clip.id) if spec.consumes_label_cases else None
        )
        if spec.consumes_label_cases and visual_payload is None:
            continue
        text = spec.clip_input(clip, mir_by_clip.get(clip.id), visual_payload)
        if text is None:
            continue
        m.clips.append(ClipCandidate(clip_id=clip.id, warning_count=0, payload=text))


def _load_candidates(
    session: Session,
    *,
    spec: LabelCaseSpec,
    labels: LabelsSettings,
) -> dict[int, list[ClipCandidate]]:
    members_by_cluster, member_by_user = _collect_members(session, case=spec.name)
    if not member_by_user:
        return {}
    if spec.runs_clip_pass:
        _populate_from_stage1(session, case=spec.name, member_by_user=member_by_user)
    else:
        _populate_from_raw_signals(session, spec=spec, member_by_user=member_by_user)
    prompt_overhead = estimate_tokens(prompt_for_cluster(labels, case=spec.name))
    out: dict[int, list[ClipCandidate]] = {}
    for cid, members in members_by_cluster.items():
        out[cid] = pick_clips(
            members,
            prompt_overhead_tokens=prompt_overhead,
            max_per_user=labels.cluster_max_clips_per_user,
            max_clips_total=labels.cluster_max_clips_per_cluster,
            token_budget=labels.cluster_sample_token_budget,
        )
    return out


class _ClusterRequest:
    """One pending cluster's batched-generation request."""

    __slots__ = ("candidates", "cluster_id", "prompt", "seed")

    def __init__(self, cluster_id: int, prompt: str, seed: int, candidates):
        self.cluster_id = cluster_id
        self.prompt = prompt
        self.seed = seed
        self.candidates = candidates


def _prepare_request(
    session: Session,
    *,
    case: str,
    cluster_id: int,
    candidates: list[ClipCandidate],
    labels: LabelsSettings,
) -> _ClusterRequest | None:
    """Build a generation request for one pending cluster.

    Clusters with no usable clips are written as a terminal ``no_input``
    failure and return ``None`` (excluded from the batch).
    """
    key = (case, cluster_id)
    if not candidates:
        upsert_terminal_failure(
            session,
            ClusterLabel,
            key=key,
            error="no_input",
            attempts=labels.cluster_max_attempts,
        )
        return None

    # Per-attempt seed variation: attempt N uses ``generation_seed + N - 1``.
    # First call uses the configured base seed; subsequent retries shift the
    # seed so validation hard fails get a real second/third chance — same
    # prompt + new seed → different sampled path → different output.
    existing = session.get(ClusterLabel, key)
    prev_attempts = (existing.attempts if existing else 0) or 0
    seed = labels.generation_seed + prev_attempts
    prompt = (
        prompt_for_cluster(labels, case=case) + "\n\n" + render_prompt_body(candidates)
    )
    return _ClusterRequest(cluster_id, prompt, seed, candidates)


def _store_result(
    session: Session,
    *,
    case: str,
    req: _ClusterRequest,
    raw: str | None,
    error: str | None,
    labels: LabelsSettings,
) -> None:
    """Validate one cluster's raw output and persist success or bump failure.

    ``error`` is set (and ``raw`` is ``None``) when generation itself raised
    for this request; otherwise ``raw`` is validated.
    """
    key = (case, req.cluster_id)
    if error is not None:
        bump_failure(
            session,
            ClusterLabel,
            key=key,
            error=f"runtime:{error}",
            max_attempts=labels.cluster_max_attempts,
            generation_seed=req.seed,
        )
        return
    assert raw is not None
    payload, status, warnings = validate_cluster(raw, labels, case=case)
    if status == "failed":
        # Validation hard fails (HC1..HC6) go through ``bump_failure`` because
        # retrying with a different seed CAN change the output.
        code = warnings[0] if warnings else "validation"
        bump_failure(
            session,
            ClusterLabel,
            key=key,
            error=format_failure_error(code, raw),
            max_attempts=labels.cluster_max_attempts,
            generation_seed=req.seed,
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
        generation_seed=req.seed,
    )
    row = session.get(ClusterLabel, key)
    assert row is not None
    row.sampled_clip_ids = [c.clip_id for c in req.candidates]


def _norm_label(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _success_label_by_cid(session: Session, case: str) -> dict[int, str]:
    rows = (
        session.execute(
            select(ClusterLabel).where(
                ClusterLabel.embedding_case == case,
                ClusterLabel.status == "success",
            )
        )
        .scalars()
        .all()
    )
    return {r.cluster_id: (r.payload or {}).get("cluster_label", "") for r in rows}


def _duplicate_extra_cids(label_by_cid: dict[int, str]) -> list[int]:
    """Cluster ids colliding on a normalized label, minus the lowest id per group."""
    groups: dict[str, list[int]] = defaultdict(list)
    for cid, lab in label_by_cid.items():
        groups[_norm_label(lab)].append(cid)
    extras: list[int] = []
    for ids in groups.values():
        if len(ids) > 1:
            extras.extend(sorted(ids)[1:])
    return sorted(extras)


def _avoid_clause(used_labels: list[str]) -> str:
    listing = "; ".join(sorted(used_labels))
    return (
        "\n\nUNIQUENESS: other clusters in this run already use the names below. "
        "Choose a DISTINCT, more specific cluster_label; do NOT reuse any of "
        f"these: {listing}."
    )


def _with_suffix(base: str, suffix: str) -> str:
    """Append ``suffix`` to ``base``, truncating the BASE (never the suffix) to
    fit ``_CLUSTER_LABEL_MAX_CHARS``.

    Slicing the whole string after concatenation would chop the disambiguating
    suffix off a near-cap base and re-collide; reserving room for the suffix
    keeps it intact so the fallback stays distinct.
    """
    room = _CLUSTER_LABEL_MAX_CHARS - len(suffix)
    if room <= 0:
        return suffix[:_CLUSTER_LABEL_MAX_CHARS]
    return f"{base[:room].rstrip()}{suffix}"


def _deterministic_distinct_label(
    row: ClusterLabel, taken: set[str], repertoire_key: str
) -> str:
    """Guaranteed-unique fallback: append the dominant tag, else the cluster id."""
    payload = row.payload or {}
    base = payload.get("cluster_label") or "Cluster"
    rep = payload.get(repertoire_key) or []
    id_label = _with_suffix(base, f" ({row.cluster_id})")
    candidates: list[str] = []
    if rep and isinstance(rep[0], dict) and rep[0].get("tag"):
        candidates.append(_with_suffix(base, f" — {rep[0]['tag']}"))
    candidates.append(id_label)
    for cand in candidates:
        if _norm_label(cand) not in taken:
            return cand
    # The cluster id is unique to this row, so the id suffix (kept intact by
    # _with_suffix) yields a label no other cluster can have produced.
    return id_label


def _disambiguate_duplicate_labels(
    session: Session,
    *,
    case: str,
    labels: LabelsSettings,
    generator,
    schema: dict,
    candidates_per_cluster: dict[int, list[ClipCandidate]],
) -> None:
    """Ensure cluster_labels are unique within ``case``.

    Clusters are labelled in isolation, so similar clusters can collide on a
    generic name. After per-case generation completes, regenerate the colliding
    clusters (keeping the lowest id of each group) with the already-used names
    injected into the prompt, for up to ``cluster_dedup_max_rounds`` rounds; any
    residual collision is resolved by a deterministic distinct suffix so
    uniqueness is guaranteed.
    """
    repertoire_key = REGISTRY[case].repertoire_key
    for round_i in range(labels.cluster_dedup_max_rounds):
        if not _dedup_round(
            session,
            case=case,
            labels=labels,
            generator=generator,
            schema=schema,
            candidates_per_cluster=candidates_per_cluster,
            round_i=round_i,
        ):
            break

    _resolve_residual_collisions(session, case=case, repertoire_key=repertoire_key)


def _build_dedup_requests(
    *,
    case: str,
    labels: LabelsSettings,
    extras: list[int],
    avoid: list[str],
    candidates_per_cluster: dict[int, list[ClipCandidate]],
) -> list[tuple[int, str, list[ClipCandidate]]]:
    reqs: list[tuple[int, str, list[ClipCandidate]]] = []
    for cid in extras:
        cands = candidates_per_cluster.get(cid, [])
        if not cands:
            continue
        prompt = (
            prompt_for_cluster(labels, case=case)
            + "\n\n"
            + render_prompt_body(cands)
            + _avoid_clause(avoid)
        )
        reqs.append((cid, prompt, cands))
    return reqs


def _dedup_round(
    session: Session,
    *,
    case: str,
    labels: LabelsSettings,
    generator,
    schema: dict,
    candidates_per_cluster: dict[int, list[ClipCandidate]],
    round_i: int,
) -> bool:
    """Run one regeneration round. Returns ``True`` to continue looping,
    ``False`` to stop (nothing to do, no requests, or generator failure)."""
    label_by_cid = _success_label_by_cid(session, case)
    extras = _duplicate_extra_cids(label_by_cid)
    if not extras:
        return False
    used = {_norm_label(v) for v in label_by_cid.values()}
    avoid = sorted(set(label_by_cid.values()))
    reqs = _build_dedup_requests(
        case=case,
        labels=labels,
        extras=extras,
        avoid=avoid,
        candidates_per_cluster=candidates_per_cluster,
    )
    if not reqs:
        return False
    # Dedicated seed space past the per-attempt range; higher temperature
    # to push the regenerated names away from the colliding originals.
    seed = labels.generation_seed + labels.cluster_max_attempts + 1 + round_i
    try:
        raws = generator.run_text_batch(
            [p for _, p, _ in reqs],
            max_new_tokens=labels.cluster_max_new_tokens,
            seeds=[seed] * len(reqs),
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            schema=schema,
        )
    except Exception:
        return False
    for (cid, _, cands), raw in zip(reqs, raws, strict=True):
        _apply_dedup_result(
            session,
            case=case,
            labels=labels,
            cid=cid,
            cands=cands,
            raw=raw,
            seed=seed,
            used=used,
        )
    generator.reclaim_memory()
    return True


def _apply_dedup_result(
    session: Session,
    *,
    case: str,
    labels: LabelsSettings,
    cid: int,
    cands: list[ClipCandidate],
    raw: str,
    seed: int,
    used: set[str],
) -> None:
    """Validate a regenerated label and persist it if it is a new unique name."""
    payload, status, _warnings = validate_cluster(raw, labels, case=case)
    if payload is None:
        return
    new_norm = _norm_label(payload.get("cluster_label", ""))
    if not new_norm or new_norm in used:
        return
    with item("WRITE", f"{case}/{cid}"):
        upsert_success(
            session,
            ClusterLabel,
            key=(case, cid),
            validation=status,
            payload=payload,
            warnings=_warnings,
            generation_seed=seed,
        )
        row = session.get(ClusterLabel, (case, cid))
        row.sampled_clip_ids = [c.clip_id for c in cands]
        session.commit()
    used.add(new_norm)


def _resolve_residual_collisions(
    session: Session, *, case: str, repertoire_key: str
) -> None:
    """Deterministic fallback for any collisions left after regeneration."""
    label_by_cid = _success_label_by_cid(session, case)
    extras = _duplicate_extra_cids(label_by_cid)
    if not extras:
        return
    taken = {_norm_label(v) for cid, v in label_by_cid.items() if cid not in extras}
    for cid in extras:
        row = session.get(ClusterLabel, (case, cid))
        new = _deterministic_distinct_label(row, taken, repertoire_key)
        row.payload = {**(row.payload or {}), "cluster_label": new}
        taken.add(_norm_label(new))
    session.commit()


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


def _gate_case(session: Session, *, case: str, current) -> None:
    """Wipe-on-drift and fingerprint gate for a case before generation."""

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


def _run_cluster_round(
    session: Session,
    *,
    case: str,
    labels: LabelsSettings,
    generator,
    schema: dict,
    pending: list[int],
    candidates_per_cluster: dict[int, list[ClipCandidate]],
) -> None:
    """One batched generate/validate/store round over the pending clusters."""
    requests: list[_ClusterRequest] = []
    for cid in pending:
        req = _prepare_request(
            session,
            case=case,
            cluster_id=cid,
            candidates=candidates_per_cluster.get(cid, []),
            labels=labels,
        )
        if req is not None:
            requests.append(req)
    session.commit()  # persist any no_input terminal failures
    if not requests:
        return

    try:
        raws = generator.run_text_batch(
            [r.prompt for r in requests],
            max_new_tokens=labels.cluster_max_new_tokens,
            seeds=[r.seed for r in requests],
            # Nucleus sampling so per-attempt seed variation produces
            # different outputs; mild temperature keeps each generation
            # close to greedy while letting retries diverge.
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            schema=schema,
        )
        errors: list[str | None] = [None] * len(requests)
    except Exception as exc:  # whole-batch failure (e.g. OOM) → bump all
        raws = [None] * len(requests)
        errors = [str(exc)] * len(requests)

    for req, raw, err in zip(requests, raws, errors, strict=True):
        with item("WRITE", f"{case}/{req.cluster_id}"):
            _store_result(
                session, case=case, req=req, raw=raw, error=err, labels=labels
            )
            session.commit()
    generator.reclaim_memory()


def _run_case(
    *,
    session: Session,
    case: str,
    labels: LabelsSettings,
    generator,
) -> None:
    spec = REGISTRY[case]
    candidates_per_cluster = _load_candidates(session, spec=spec, labels=labels)
    current = _fingerprint_for(
        session,
        spec=spec,
        labels=labels,
        candidates_per_cluster=candidates_per_cluster,
    )

    _gate_case(session, case=case, current=current)

    all_ids = sorted(candidates_per_cluster)
    schema = cluster_schema(REGISTRY[case], labels)
    # Loop until every cluster reaches a terminal state (success or
    # ``status='failed'`` with attempts == cluster_max_attempts). Each round
    # generates ALL still-pending clusters in ONE batched vLLM call (fused MoE
    # + continuous batching), then validates/stores each; failed HC rounds bump
    # the row to ``pending`` with a fresh attempt count and a new per-cluster
    # seed, re-entering this loop — one pipeline run consumes the full retry
    # budget instead of needing N restarts.
    first_iteration = True
    while True:
        pending = _pending_cluster_ids(
            session,
            case=case,
            all_ids=all_ids,
            max_attempts=labels.cluster_max_attempts,
        )
        if not pending:
            break
        if first_iteration:
            event(
                "GET",
                "qwen3-cluster",
                stats={"case": case, "clusters": len(pending)},
            )
            first_iteration = False

        _run_cluster_round(
            session,
            case=case,
            labels=labels,
            generator=generator,
            schema=schema,
            pending=pending,
            candidates_per_cluster=candidates_per_cluster,
        )

    # Enforce within-case label uniqueness before sealing.
    _disambiguate_duplicate_labels(
        session,
        case=case,
        labels=labels,
        generator=generator,
        schema=schema,
        candidates_per_cluster=candidates_per_cluster,
    )

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
