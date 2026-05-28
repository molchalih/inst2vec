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
from core.log import stage
from modules.embeddings.cases import default_cases
from modules.labels import clip_pass, cluster_pass
from modules.labels.cases import REGISTRY
from modules.labels.gc import purge_orphans
from modules.labels.models import LabelsGenerator
from modules.labels.ordering import case_run_order


@stage("labels")
def run(settings: Settings, secrets: Secrets) -> None:
    """Run orphan GC → per-case stage-1 → per-case stage-2 cluster pass."""
    del secrets  # unused — labels uses no external credentials.
    labels = settings.labels
    generator = LabelsGenerator.lazy(labels)
    try:
        with Session(get_engine()) as session:
            purge_orphans(session)
            session.commit()

        cases = tuple(case_run_order(default_cases(settings), registry=REGISTRY))
        for case in cases:
            with Session(get_engine()) as session:
                clip_pass.run_case(
                    session=session,
                    settings=settings,
                    labels=labels,
                    generator=generator,
                    spec=REGISTRY[case],
                )

        with Session(get_engine()) as session:
            cluster_pass.run_all_cases(
                session=session,
                labels=labels,
                generator=generator,
                cases=cases,
            )
    finally:
        generator.unload()
