from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, field_validator

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


class PathsSettings(BaseModel):
    video_dir: str
    plots_dir: str
    model_path: str
    profile_pic_dir: str
    thumbnail_dir: str
    speech_audio_dir: str
    data_csv_path: str


class ParseSettings(BaseModel):
    fetch_retry_delays_sec: list[int]


class DownloadSettings(BaseModel):
    max_attempts: int
    retry_delay: int
    retry_jitter: int
    concurrency: int


class FilterSettings(BaseModel):
    min_video_duration: int = 3
    max_video_duration: int = 80
    min_taken_at: int = 1640995200
    creator_min_median_views: int = 10000
    min_eligible_clips_per_user: int = 10
    global_low_percentile: float = 5
    global_high_percentile: float = 99
    creator_low_z_threshold: float = -3.5
    selection_pool_percent: float = 0.20
    selected_clips_per_user: int = 10
    selection_random_seed: int = 42


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
    api_max_attempts: int
    api_retry_delay: float
    api_retry_jitter: float
    acr_max_attempts: int
    ffmpeg_timeout_seconds: int

    @field_validator(
        "commit_every", "api_max_attempts", "acr_max_attempts", "ffmpeg_timeout_seconds"
    )
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


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
    vad_enabled: bool
    vad_sampling_rate: int
    vad_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int
    vad_speech_pad_ms: int
    vad_min_total_speech_s: float


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
    paths: PathsSettings
    parse: ParseSettings
    download: DownloadSettings
    filter: FilterSettings
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
        paths=PathsSettings(**raw["paths"]),
        parse=ParseSettings(**raw["parse"]),
        download=DownloadSettings(**raw["download"]),
        filter=FilterSettings(**raw.get("filter", {})),
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
