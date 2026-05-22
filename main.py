from core.splash import boot

boot()

# ruff: noqa: E402
from collections.abc import Callable

from core.config import Secrets, Settings, load_runtime_config
from core.console import phase, pipeline
from core.database import init_db
from modules import (
    captions,
    clustering,
    embeddings,
    filter,
    ingest,
    mir,
    speech,
    upload,
)
from modules.embeddings.cases import default_cases


def _init_db_stage(_settings: Settings, secrets: Secrets) -> None:
    init_db(secrets.database_url, secrets.identity_db_url)


# Pipeline DAG (upstream → downstream, by Fingerprint.dependency edges):
#   Database             : —
#   Importing            : Database
#   Profile Parsing      : Importing
#   Processing Dataset   : Profile Parsing
#   Download             : Processing Dataset
#   Upload               : Download
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
        ("Upload", upload.run),
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
    ]


def run_pipeline() -> None:
    settings, secrets = load_runtime_config()
    cases = default_cases(settings)
    stages = _stages(cases)
    with pipeline(total_stages=len(stages)):
        for name, fn in stages:
            phase(name)
            fn(settings, secrets)


if __name__ == "__main__":
    run_pipeline()
