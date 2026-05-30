"""Multimodal labels stage: per-case stage-1 clip-pass + cluster pass.

Orchestrator only. The heavy lifting lives in:

- :mod:`modules.labels.clip_pass` — generic per-case stage-1 runner.
- :mod:`modules.labels.cluster_pass` — per-case stage-2 cluster synthesis.

The body loops over ``default_cases(settings)`` dispatching to the generic
runners; case-specific behaviour flows exclusively through the
``LabelCaseSpec`` registry (no ``if case == "..."`` branches here).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from core.config import Secrets, Settings
from core.database.engine import get_engine
from core.gpu_memory import oom_guard, reset_peak, vram_fields
from core.log import event, stage
from modules.embeddings.cases import default_cases
from modules.labels import clip_pass, cluster_pass
from modules.labels.cases import REGISTRY
from modules.labels.gc import purge_orphans
from modules.labels.models import ClusterLabelsGenerator, LabelsGenerator
from modules.labels.ordering import case_run_order


def _emit_vram(at: str) -> None:
    """Emit a visible VRAM line within the active ``labels`` scope.

    No-op without CUDA (``vram_fields`` returns ``None``). Must be called from
    inside the ``@stage("labels")`` scope so ``event`` has a scope to render in.
    """
    fields = vram_fields(at=at)
    if fields is not None:
        event("LOAD", "vram", stats=fields)


@stage("labels")
def run(settings: Settings, secrets: Secrets) -> None:
    """Run orphan GC → per-case stage-1 → per-case stage-2 cluster pass."""
    del secrets  # unused — labels uses no external credentials.
    labels = settings.labels
    generator = LabelsGenerator.lazy(labels)
    cluster_generator = None
    try:
        with Session(get_engine()) as session:
            purge_orphans(session)
            session.commit()

        cases = tuple(case_run_order(default_cases(settings), registry=REGISTRY))
        # Stage 1 only runs for cases that need a per-clip Qwen pass. The
        # rephrasing cases (sandwich/auditory/spoken/textual) skip stage 1 entirely
        # and let the cluster pass synthesise from raw caption/speech/MIR
        # signals — for those cases ``ClipLabel`` rows are dead weight and
        # are purged by ``purge_orphans`` above.
        for case in cases:
            spec = REGISTRY[case]
            if not spec.runs_clip_pass:
                continue
            with Session(get_engine()) as session:
                clip_pass.run_case(
                    session=session,
                    settings=settings,
                    labels=labels,
                    generator=generator,
                    spec=spec,
                )

        # Free the VL-8B weights before the 30B Int4 cluster model loads —
        # the 24 GB card cannot hold both. unload() is idempotent; the
        # finally block repeats it harmlessly.
        _emit_vram("before-vl-unload")
        with oom_guard("VL-8B unload"):
            generator.unload()
        _emit_vram("after-vl-unload")
        reset_peak()

        cluster_generator = ClusterLabelsGenerator.lazy(labels)
        with oom_guard("cluster pass (30B Int4)"), Session(get_engine()) as session:
            cluster_pass.run_all_cases(
                session=session,
                labels=labels,
                generator=cluster_generator,
                cases=cases,
            )
        _emit_vram("after-cluster-pass")
    finally:
        generator.unload()
        if cluster_generator is not None:
            cluster_generator.unload()
