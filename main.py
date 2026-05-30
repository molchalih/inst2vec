from core.splash import boot

boot()

# ruff: noqa: E402
import argparse
import os
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
    upload,
    visualization,
)
from modules.embeddings.cases import default_cases


@stage("db")
def _init_db_stage(settings: Settings, secrets: Secrets) -> None:
    init_db(secrets.database_url, secrets.identity_db_url)
    # Self-healing, idempotent: re-key legacy maest→auditory (no recompute)
    # and drop legacy audio rows. No-op on a clean tree.
    from core.database.case_migration import run_case_migration_at_startup

    run_case_migration_at_startup(settings)


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


def cli() -> None:
    parser = argparse.ArgumentParser(prog="inst2vec")
    parser.add_argument("--pod", action="store_true", help="run as an embedding pod")
    parser.add_argument("--host", default="", help="orchestrator host:port (pod mode)")
    parser.add_argument(
        "--video-root",
        default=os.environ.get("VIDEO_ROOT", "/workspace/videos"),
        help="mounted video directory (pod mode)",
    )
    args = parser.parse_args()
    if args.pod:
        if not args.host:
            raise SystemExit("--pod requires --host=<orchestrator:port>")
        from modules.embeddings.pod import run_pod

        run_pod(args.host, args.video_root)
        return
    run_pipeline()


if __name__ == "__main__":
    cli()
