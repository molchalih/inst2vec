from __future__ import annotations

import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


def load_runtime_config() -> tuple[SimpleNamespace, SimpleNamespace]:
    with open(_CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)

    settings = SimpleNamespace(
        pipeline=SimpleNamespace(**raw["pipeline"]),
        paths=SimpleNamespace(**raw["paths"]),
        parse=SimpleNamespace(**raw["parse"]),
        download=SimpleNamespace(**raw["download"]),
        finalize=SimpleNamespace(**raw["finalize"]),
        music=SimpleNamespace(**raw["music"]),
        speech=SimpleNamespace(**raw["speech"]),
        captions=SimpleNamespace(**raw["captions"]),
        embeddings=SimpleNamespace(**raw["embeddings"]),
        search=SimpleNamespace(**raw.get("search", {})),
        validation=SimpleNamespace(**raw["validation"]),
        overrides=SimpleNamespace(**raw["overrides"]),
    )

    secrets = SimpleNamespace(
        database_url=os.environ["DATABASE_URL"],
        identity_db_url=os.environ["IDENTITY_DB_URL"],
        hiker_api_key=os.environ["HIKER_API_KEY"],
        arc_host=os.environ["ARC_HOST"],
        arc_access_key=os.environ["ARC_ACCESS_KEY"],
        arc_secret_key=os.environ["ARC_SECRET_KEY"],
        spotify_client_id=os.environ["SPOTIFY_CLIENT_ID"],
        spotify_client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
        huggingface_token=os.environ["HUGGINGFACE_TOKEN"],
    )

    return settings, secrets
