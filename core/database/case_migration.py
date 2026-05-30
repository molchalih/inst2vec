"""One-off, idempotent embedding-case migration.

Re-keys the legacy ``maest`` case to ``auditory`` WITHOUT recomputing any
stored embedding / clustering / label / visualization data, and removes the
legacy ``audio`` case entirely (it is replaced by the speech-only ``spoken``
case, a deliberate full recompute).

Why seal adoption is required (and why it must be GATED)
--------------------------------------------------------
``modules.embeddings.cases.case_config_identity`` folds ``case={spec.name}``
into the embed config-identity string, and the labels cluster-pass config
payload folds ``cluster_case_prompts.{case}=`` — so renaming the case
perturbs those two config hashes. A naive ``scope_key`` rename would leave a
stale ``config_hash`` and the next pipeline run would treat the renamed case
as drifted and WIPE + recompute. To preserve the data, the migration ADOPTS
the post-rename config hashes (computed by the caller from the renamed
registry/config) into the migrated seals.

Adopting the auditory hash is only correct when the *sole* difference between
the stored maest seal and the current recipe is the case name. If the MAEST
checkpoint / sample-rate / aggregation settings (embed) or the cluster-label
prompt / generator knobs (cluster_labels) drifted before this migration ran,
the stored maest data is itself stale and MUST be recomputed. So adoption is
gated: the caller also supplies the *expected legacy* hash — the hash the OLD
``maest`` case would produce under CURRENT settings (identical recipe, only
``case=maest``). The auditory seal is adopted ONLY when the stored maest seal's
``config_hash`` equals that legacy hash. On mismatch the data rows are still
re-keyed (the rename is unconditional), but NO current seal is written: the old
seal is dropped so the stage recomputes for ``auditory``. The embed runner does
NOT wipe on a missing seal (it incremental-diffs per-clip ``source_hash``, which
is config-independent), so on an embed drop the migration ALSO deletes the
re-keyed ``ClipEmbedding`` rows to force a full re-embed; the cluster-labels
runner self-wipes on a missing seal, so its re-keyed rows are left in place.

The visualization seal's ``dependency_hash`` chains to the cluster_assign +
cluster_labels seals; because the cluster_labels config hash may be adopted
(and thus change), the visualization dependency hash is re-derived from the
adopted upstream rows so visualization also stays frozen (SKIP). When an
upstream seal was NOT adopted (config drift), its auditory row is absent, so
the re-derived dependency hash naturally differs and visualization recomputes
too.

Idempotent: guarded on "any maest/audio row still exists", so a clean
(already-migrated) database is a no-op, and a partially-migrated database
converges.
"""

from __future__ import annotations

from sqlalchemy import update
from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.database.models import (
    ClipEmbedding,
    ClipLabel,
    ClusterLabel,
    ClusterRun,
    StageState,
    UserCluster,
    UserEmbedding,
    Visualization,
    VisualizationCluster,
    VisualizationUser,
)

_OLD = "maest"
_NEW = "auditory"
_DROP = "audio"

# (model, case-column-name) for every table carrying the case string.
_CASE_TABLES: tuple[tuple[type, str], ...] = (
    (ClipEmbedding, "embedding_case"),
    (UserEmbedding, "embedding_case"),
    (UserCluster, "embedding_case"),
    (ClusterRun, "embedding_case"),
    (Visualization, "embedding_case"),
    (VisualizationUser, "embedding_case"),
    (VisualizationCluster, "embedding_case"),
    (ClipLabel, "label_case"),
    (ClusterLabel, "embedding_case"),
)

# Stage names whose StageState rows are scoped by the raw case string.
_EMBED_STAGE = "clip_embeddings"
_CLUSTER_LABELS_STAGE = "cluster_labels"
_CLUSTER_ASSIGN_STAGE = "cluster_assign"
_VIZ_STAGE = "visualization"
# Every case-scoped seal stage. ``labels`` (clip pass) is included for safety:
# the acoustic case skips the clip pass today so it usually has no such row,
# but re-keying one if present is harmless and keeps the migration complete.
_CASE_SCOPED_STAGES: tuple[str, ...] = (
    _EMBED_STAGE,
    "user_embeddings",
    "cluster_search",
    "cluster_validation",
    _CLUSTER_ASSIGN_STAGE,
    "labels",
    _CLUSTER_LABELS_STAGE,
    _VIZ_STAGE,
)


def legacy_cluster_labels_payload(auditory_payload: str) -> str:
    """Reconstruct the OLD maest cluster-labels config payload from the auditory one.

    The maest→auditory rename moved the SAME cluster prompt body from the
    ``maest`` key to the ``auditory`` key of ``cluster_case_prompts``. So
    ``cluster_labels_config_payload(settings.labels, "maest")`` now hashes an
    EMPTY prompt (the maest key is gone) and can never match the seal the old
    maest recipe produced — which would drop the seal on a normal, drift-free
    rename and force an unnecessary wipe + recompute of the auditory cluster
    labels. Mirroring the embed legacy-identity reconstruction, the legacy
    payload is the current auditory payload with only its leading
    ``cluster_case_prompts.auditory=`` key rewritten to ``...maest=``: same base
    knobs, same prompt body, only the case token differs. Any real drift in the
    base knobs or the prompt body still breaks the equality and forces recompute.
    """
    return auditory_payload.replace(
        f"cluster_case_prompts.{_NEW}=", f"cluster_case_prompts.{_OLD}=", 1
    )


def _needs_migration(session: Session) -> bool:
    for model, attr in _CASE_TABLES:
        col = getattr(model, attr)
        if session.query(model).filter(col.in_((_OLD, _DROP))).first() is not None:
            return True
    for stage in _CASE_SCOPED_STAGES:
        for case in (_OLD, _DROP):
            if session.get(StageState, (stage, case)) is not None:
                return True
    return False


def _delete_audio(session: Session) -> None:
    for model, attr in _CASE_TABLES:
        session.query(model).filter(getattr(model, attr) == _DROP).delete(
            synchronize_session=False
        )
    for stage in _CASE_SCOPED_STAGES:
        row = session.get(StageState, (stage, _DROP))
        if row is not None:
            session.delete(row)


def _rekey_data_rows(session: Session) -> None:
    for model, attr in _CASE_TABLES:
        session.execute(
            update(model).where(getattr(model, attr) == _OLD).values(**{attr: _NEW})
        )


def _rekey_seal(session: Session, stage: str) -> None:
    """Rename the case-independent seal ``(stage, maest)`` → ``(stage, auditory)``.

    For stages whose config hash does NOT fold the case name, a plain
    ``scope_key`` rename preserves a valid seal. StageState's PK includes
    ``scope_key`` so the rename is a delete + insert of a fresh row carrying the
    same hashes. Idempotent: if the old row is absent this is a no-op.
    """
    old = session.get(StageState, (stage, _OLD))
    if old is None:
        return
    session.delete(old)
    session.flush()
    session.merge(
        StageState(
            stage_name=stage,
            scope_key=_NEW,
            data_hash=old.data_hash,
            config_hash=old.config_hash,
            dependency_hash=old.dependency_hash,
        )
    )
    session.flush()


def _adopt_or_drop_seal(
    session: Session,
    stage: str,
    *,
    auditory_config_hash: str,
    legacy_config_hash: str,
) -> bool:
    """Rename ``(stage, maest)`` → ``(stage, auditory)`` for a case-name-folding seal.

    Adopts the current ``auditory_config_hash`` ONLY when the stored maest seal's
    ``config_hash`` equals ``legacy_config_hash`` (the hash the old ``maest``
    recipe produces under current settings). That equality proves nothing but the
    case name changed, so the stored data is current and may be sealed as done.

    On mismatch the stored maest data is stale (recipe drifted): the old seal is
    DELETED and no auditory seal is written, so the next pipeline run sees no
    seal and recomputes the stage for ``auditory`` via the normal drift path.

    Returns ``True`` only when a stored maest seal was actively DROPPED on
    mismatch (the re-keyed auditory data is therefore stale). Returns ``False``
    when the seal was adopted or when no maest seal existed — in the latter case
    we must NOT treat any auditory rows as stale, since they may be legitimate
    already-migrated data rather than re-keyed maest rows.

    Idempotent: if the old row is absent this is a no-op.
    """
    old = session.get(StageState, (stage, _OLD))
    if old is None:
        return False
    matched = old.config_hash == legacy_config_hash
    session.delete(old)
    session.flush()
    if matched:
        session.merge(
            StageState(
                stage_name=stage,
                scope_key=_NEW,
                data_hash=old.data_hash,
                config_hash=auditory_config_hash,
                dependency_hash=old.dependency_hash,
            )
        )
        session.flush()
    return not matched


def run_case_migration(
    session: Session,
    *,
    embed_config_hash: str,
    embed_legacy_config_hash: str,
    cluster_labels_config_hash: str,
    cluster_labels_legacy_config_hash: str,
) -> None:
    """Re-key maest→auditory (no recompute when valid) and drop legacy audio rows.

    The ``*_config_hash`` args are the post-rename (auditory) config hashes the
    caller computes from the renamed registry / config; the ``*_legacy_config_hash``
    args are the hashes the OLD ``maest`` recipe produces under CURRENT settings
    (identical recipe, ``case=maest``). For each case-name-folding seal the
    auditory hash is adopted ONLY when the stored maest seal matches the legacy
    hash — proving nothing but the case name drifted. On mismatch the data rows
    are still re-keyed but no auditory seal is written, so the stage recomputes
    via normal fingerprint drift.

    Caller owns the transaction boundary (this commits once at the end).
    """
    if not _needs_migration(session):
        return

    _delete_audio(session)
    _rekey_data_rows(session)

    # Seals whose config hash folds the literal case name: adopt the auditory
    # hash only when the stored maest seal matches the expected legacy hash.
    embed_dropped = _adopt_or_drop_seal(
        session,
        _EMBED_STAGE,
        auditory_config_hash=embed_config_hash,
        legacy_config_hash=embed_legacy_config_hash,
    )
    if embed_dropped:
        # The embed runner treats a missing StageState as a first run and diffs
        # only per-clip ``source_hash`` (which is config-independent); unlike the
        # cluster-pass runner it does NOT wipe on a missing seal. So the re-keyed
        # auditory rows would be sealed stale under the new config instead of
        # recomputed. Drop them here so the next run re-embeds every clip.
        session.query(ClipEmbedding).filter(
            ClipEmbedding.embedding_case == _NEW
        ).delete(synchronize_session=False)
        session.flush()
    # cluster_labels needs no data wipe on drop: its cluster-pass runner wipes
    # the case up front when no seal is present (modules/labels/cluster_pass.py).
    _adopt_or_drop_seal(
        session,
        _CLUSTER_LABELS_STAGE,
        auditory_config_hash=cluster_labels_config_hash,
        legacy_config_hash=cluster_labels_legacy_config_hash,
    )
    # The remaining case-scoped seals' hashes are case-name-independent, so a
    # plain scope_key rename keeps them valid.
    for stage in _CASE_SCOPED_STAGES:
        if stage in (_EMBED_STAGE, _CLUSTER_LABELS_STAGE, _VIZ_STAGE):
            continue
        _rekey_seal(session, stage)

    # Visualization: config is empty (case-independent) but its dependency hash
    # chains to cluster_assign + cluster_labels. Re-derive it from the auditory
    # upstream seals so viz stays frozen when both upstreams were adopted; if an
    # upstream seal was dropped (drift), its auditory row is absent and the
    # re-derived hash differs, so visualization recomputes alongside it.
    viz_old = session.get(StageState, (_VIZ_STAGE, _OLD))
    if viz_old is not None:
        new_dep = fp.compose_hashes(
            fp.stage_dependency_hash(session, _CLUSTER_ASSIGN_STAGE, _NEW),
            fp.stage_dependency_hash(session, _CLUSTER_LABELS_STAGE, _NEW),
        )
        session.delete(viz_old)
        session.flush()
        session.merge(
            StageState(
                stage_name=_VIZ_STAGE,
                scope_key=_NEW,
                data_hash=viz_old.data_hash,
                config_hash=viz_old.config_hash,
                dependency_hash=new_dep,
            )
        )

    session.commit()


def run_case_migration_at_startup(settings) -> None:
    """Self-healing entry point called once after ``init_db``.

    Computes the adopted post-rename config hashes from the renamed registry +
    config and runs :func:`run_case_migration` in its own session. A no-op on a
    clean (already-migrated) database, so it is safe on every pipeline run.
    """
    from core import fingerprint as fp
    from core.database.engine import get_session
    from modules.embeddings.cases import AUDITORY_CASE, case_config_identity
    from modules.labels.state import cluster_labels_config_payload

    session = get_session()
    try:
        # Cheap guard FIRST: on a clean (already-migrated) tree there is
        # nothing to do, and we avoid touching ``settings`` at all — keeping
        # this safe for callers with a minimal settings stub.
        if not _needs_migration(session):
            return
        # ``case_config_identity`` starts the string with ``case={spec.name}``
        # and the maest-specific recipe parts are gated on ``spec.name ==
        # "auditory"``; the legacy maest seal historically carried the same
        # recipe tail but with ``case=maest``. So the expected-legacy embed hash
        # is the auditory identity string with only its leading case token
        # rewritten — the tail (checkpoint sha, sample rate, aggregation, …) is
        # identical, so any drift in those parts breaks the equality and forces
        # recompute, exactly as required.
        auditory_identity = case_config_identity(AUDITORY_CASE, settings)
        legacy_identity = auditory_identity.replace(f"case={_NEW}", f"case={_OLD}", 1)
        embed_config_hash = fp.hash_text(auditory_identity)
        embed_legacy_config_hash = fp.hash_text(legacy_identity)
        auditory_cluster_payload = cluster_labels_config_payload(settings.labels, _NEW)
        cluster_labels_config_hash = fp.hash_text(auditory_cluster_payload)
        # Reconstruct the legacy maest payload by swapping ONLY the case token —
        # hashing ``..., _OLD)`` directly would read an empty (now-absent) maest
        # prompt and never match the real maest seal. See
        # ``legacy_cluster_labels_payload``.
        cluster_labels_legacy_config_hash = fp.hash_text(
            legacy_cluster_labels_payload(auditory_cluster_payload)
        )
        run_case_migration(
            session,
            embed_config_hash=embed_config_hash,
            embed_legacy_config_hash=embed_legacy_config_hash,
            cluster_labels_config_hash=cluster_labels_config_hash,
            cluster_labels_legacy_config_hash=cluster_labels_legacy_config_hash,
        )
    finally:
        session.close()
