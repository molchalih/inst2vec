from core.splash import boot

boot()

# ruff: noqa: E402
from collections.abc import Callable

from core.config import Secrets, Settings, load_runtime_config
from core.console import pipeline
from core.database import init_db
from core.log import stage
from modules import (
    captions,
    clustering,
    embeddings,
    filter,
    ingest,
    labels,
    mir,
    speech,
    visualization,
)
from modules.embeddings.cases import default_cases


@stage("db")
def _init_db_stage(settings: Settings, secrets: Secrets) -> None:
    init_db(secrets.database_url, secrets.identity_db_url)


# Pipeline DAG (upstream → downstream, by Fingerprint.dependency edges):
#   Database             : —
#   Importing            : Database
#   Profile Parsing      : Importing
#   Processing Dataset   : Profile Parsing
#   Download             : Processing Dataset
#   Audio extraction     : Download
#   MIR audio extraction : Audio extraction
#   MIR inference        : MIR audio extraction
#   Speech               : Audio extraction
#   Captions             : Profile Parsing
#   Clip Embeddings      : Speech, Captions, MIR (case-dependent)
#   User Embeddings      : Clip Embeddings
#   Cluster Search       : User Embeddings
#   Cluster Validation   : Cluster Search
#   Clustering           : Cluster Validation
#   Visual Labels        : Processing Dataset (selected clips only)
#   Visualization        : Clustering
# Reorder = noisy fingerprint resets on the next run, not data loss.
def _stages(
    cases: tuple[str, ...],
) -> list[tuple[str, Callable[[Settings, Secrets], None]]]:
    return [
        ("Database", _init_db_stage),
        ("Importing", ingest.run_seed),
        ("Profile Parsing", ingest.run_profiles),
        ("Processing Dataset", filter.run),
        ("Download", ingest.run_download),
        ("Audio extraction", ingest.run_audio),
        ("MIR audio extraction", ingest.run_audio_mir),
        ("MIR inference", mir.run_mir),
        ("Speech transcription", speech.run),
        ("Captions translation", captions.run),
        ("Clip Embeddings", embeddings.run_clip),
        ("User Embeddings", embeddings.run_users),
        ("Cluster Search", lambda s, x: clustering.run_search(s, x, cases)),
        ("Cluster Validation", lambda s, x: clustering.run_validation(s, x, cases)),
        ("Clustering", lambda s, x: clustering.run_assign(s, x, cases)),
        ("Visual Labels", labels.run),
        ("Visualization", lambda s, x: visualization.run(s, x, cases)),
    ]


def run_pipeline() -> None:
    settings, secrets = load_runtime_config()
    cases = default_cases(settings)
    stages = _stages(cases)
    with pipeline(total_stages=len(stages)):
        for _name, fn in stages:
            fn(settings, secrets)


if __name__ == "__main__":
    run_pipeline()
