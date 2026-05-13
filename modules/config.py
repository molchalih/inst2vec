from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


class PipelineSettings(BaseModel):
    batch_size: int
    max_clips: int


class PathsSettings(BaseModel):
    video_dir: str
    plots_dir: str
    model_path: str
    profile_pic_dir: str
    thumbnail_dir: str
    data_csv_path: str


class ParseSettings(BaseModel):
    fetch_retry_delays_sec: list[int]


class DownloadSettings(BaseModel):
    max_attempts: int
    retry_delay: int


class FinalizeSettings(BaseModel):
    target_clips_per_user: int
    require_min_text_clips: bool
    pass_a_recompute_from_scratch: bool
    global_min_plays: int
    global_min_plays_percentile: float
    creator_robust_z_threshold: float
    creator_min_clips: int


class MusicSettings(BaseModel):
    audio_fingerprint_confidence: float
    commit_every: int
    http_timeout: float
    spotify_search_limit: int
    spotify_token_skew_seconds: int
    spotify_request_timeout: float
    reccobeats_batch_size: int
    reccobeats_delay_min: float
    reccobeats_delay_max: float
    manual_features_max_seconds: int
    manual_features_sample_rate: int
    manual_features_max_mb: float
    manual_features_mp3_bitrate: str


class SpeechSettings(BaseModel):
    whisper_model: str
    commit_every: int
    translate_model: str
    translate_target_lang: str
    translation_max_chars: int
    translate_max_new_tokens: int
    logprob_threshold: float
    compression_threshold: float
    min_meaningful_chars: int


class CaptionsSettings(BaseModel):
    commit_every: int
    translate_model: str
    translate_target_lang: str
    translation_max_chars: int
    translate_max_new_tokens: int


class EmbeddingsSettings(BaseModel):
    exclude_disqualified_users: bool
    embed_max_length: int
    adaptive_max_frames: int
    adaptive_default_fps: float


class SearchSettings(BaseModel):
    umap_n_components: list[int] = []
    umap_n_neighbors: list[int] = []
    umap_min_dist: list[float] = []
    umap_metrics: list[str] = []
    umap2d_n_neighbors: int = 15
    umap2d_min_dist: float = 0.1
    umap2d_metrics: list[str] = []
    hdbscan_min_cluster_size: list[int] = []
    hdbscan_selection: list[str] = []
    random_state: int = 42
    clustering_grid_workers: int = 1


class ValidationSettings(BaseModel):
    plateau_drop_threshold: float
    max_noise_ratio: float
    min_clusters: int
    max_clusters: int


class OverridesSettings(BaseModel):
    video: str = ""
    sandwich: str = ""
    audio: str = ""


class Settings(BaseModel):
    pipeline: PipelineSettings
    paths: PathsSettings
    parse: ParseSettings
    download: DownloadSettings
    finalize: FinalizeSettings
    music: MusicSettings
    speech: SpeechSettings
    captions: CaptionsSettings
    embeddings: EmbeddingsSettings
    search: SearchSettings
    validation: ValidationSettings
    overrides: OverridesSettings


class Secrets(BaseModel):
    database_url: str
    identity_db_url: str
    hiker_api_key: str
    arc_host: str
    arc_access_key: str
    arc_secret_key: str
    spotify_client_id: str
    spotify_client_secret: str
    huggingface_token: str


def load_runtime_config() -> tuple[Settings, Secrets]:
    with open(_CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)

    settings = Settings(
        pipeline=PipelineSettings(**raw["pipeline"]),
        paths=PathsSettings(**raw["paths"]),
        parse=ParseSettings(**raw["parse"]),
        download=DownloadSettings(**raw["download"]),
        finalize=FinalizeSettings(**raw["finalize"]),
        music=MusicSettings(**raw["music"]),
        speech=SpeechSettings(**raw["speech"]),
        captions=CaptionsSettings(**raw["captions"]),
        embeddings=EmbeddingsSettings(**raw["embeddings"]),
        search=SearchSettings(**raw.get("search", {})),
        validation=ValidationSettings(**raw["validation"]),
        overrides=OverridesSettings(**raw["overrides"]),
    )

    secrets = Secrets(
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
