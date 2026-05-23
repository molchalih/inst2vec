from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


class PathsSettings(BaseModel):
    video_dir: str
    model_path: str
    profile_pic_dir: str
    thumbnail_dir: str
    speech_audio_dir: str
    # audio_dir is always created and populated by extract_audio_stage;
    # downstream stages assume it exists when audio was extracted.
    audio_dir: str = "data/audio"
    audio_mir_dir: str = "data/audio_mir"
    data_csv_path: str

    def video_for(self, clip_id: int) -> Path:
        return Path(self.video_dir) / f"{clip_id}.mp4"

    def audio_for(self, clip_id: int) -> Path:
        return Path(self.audio_dir) / f"{clip_id}.mp3"

    def audio_mir_for(self, clip_id: int) -> Path:
        return Path(self.audio_mir_dir) / f"{clip_id}.wav"

    def thumbnail_for(self, clip_id: int) -> Path:
        return Path(self.thumbnail_dir) / f"{clip_id}.jpg"


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
    min_eligible_clips_per_user: int = 30
    global_low_percentile: float = 5
    global_high_percentile: float = 99
    creator_low_z_threshold: float = -3.5
    selection_pool_percent: float = 0.20
    selected_clips_per_user: int = 10
    selection_random_seed: int = 42


class MirSettings(BaseModel):
    binary_threshold: float = 0.5
    music_min_confidence: float = 0.30
    music_min_margin: float = 0.05
    topk_genre: int = 10
    topk_moodtheme: int = 10
    topk_instrument: int = 10
    inference_sample_rate: int = 16_000
    model_dir: str = "models/mir"
    download_concurrency: int = 4
    commit_every: int = 25
    prefetch_queue_size: int = 2
    http_timeout: float = 30.0

    maest_checkpoint: str = "discogs-maest-30s-pw-519l-1.pb"
    maest_input: str = "serving_default_melspectrogram"
    maest_output: str = "StatefulPartitionedCall:0"
    maest_patch_seconds: float = 30.0
    effnet_checkpoint: str = "discogs-effnet-bs64-1.pb"
    effnet_embed_output: str = "PartitionedCall:1"

    checkpoint_max_attempts: int = 3
    checkpoint_backoff_seconds: float = 2.0

    @field_validator(
        "topk_genre",
        "topk_moodtheme",
        "topk_instrument",
        "inference_sample_rate",
        "download_concurrency",
        "commit_every",
        "prefetch_queue_size",
        "checkpoint_max_attempts",
    )
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("binary_threshold")
    @classmethod
    def _ratio(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("must lie in (0, 1)")
        return v

    @field_validator("music_min_confidence", "music_min_margin")
    @classmethod
    def _unit_interval(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("must lie in [0, 1]")
        return v

    @field_validator("maest_patch_seconds")
    @classmethod
    def _positive_float(cls, v: float) -> float:
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
    dirty_min_chars: int
    dirty_min_letter_ratio: float
    vad_enabled: bool
    vad_sampling_rate: int
    vad_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int
    vad_speech_pad_ms: int
    vad_min_total_speech_s: float
    vad_ffmpeg_timeout_s: int


class CaptionsSettings(BaseModel):
    commit_every: int
    translate_model: str
    translate_target_lang: str
    translation_max_chars: int
    translate_max_new_tokens: int


class AudioExtractionSettings(BaseModel):
    audio_bitrate_kbps: int = 128
    audio_sample_rate_hz: int = 44100
    audio_extract_timeout_s: int = 60
    mir_codec: str = "pcm_s16le"
    mir_extension: str = "wav"
    mir_sample_rate_hz: int = 16_000
    mir_channels: int = 1
    mir_extract_timeout_s: int = 60


class EmbeddingsSettings(BaseModel):
    exclude_disqualified_users: bool
    embed_max_length: int
    adaptive_max_frames: int
    adaptive_default_fps: float
    provider: Literal["local", "remote"] = "local"
    inflight: int = 1
    request_timeout_s: int = 120
    max_retries: int = 3
    # ── Gemini Embedding 2 case ──
    gemini_enabled: bool = False
    gemini_model: str = "gemini-embedding-2-preview"
    gemini_output_dim: int = 3072
    gemini_max_video_seconds: int = 120
    gemini_max_audio_seconds: int = 80
    gemini_request_timeout_s: int = 60
    gemini_max_retries: int = 5

    @field_validator("inflight", "request_timeout_s", "max_retries")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class SearchSettings(BaseModel):
    umap_n_components: list[int] = []
    umap_n_neighbors: list[int] = []
    umap_min_dist: list[float] = []
    umap_metrics: list[str] = []
    umap2d_n_neighbors: int = 15
    umap2d_min_dist: float = 0.1
    umap2d_metrics: list[str] = []
    hdbscan_min_cluster_size: list[int] = []
    hdbscan_min_samples: list[int] = []
    hdbscan_selection: list[str] = []
    random_state: int = 42
    clustering_grid_workers: int = 1


class ValidationSettings(BaseModel):
    plateau_drop_threshold: float
    max_noise_ratio: float
    min_clusters: int
    max_clusters: int


class StorageSettings(BaseModel):
    backend: str = "s3"
    bucket: str = ""
    prefix: str = "videos/"
    signed_url_ttl_s: int = 3600

    @field_validator("signed_url_ttl_s")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class VisualizationSettings(BaseModel):
    export_dir: Path
    default_case: str
    distinctiveness_z_min: float = 0.5
    distinctiveness_top_k: int = 3
    genre_top_k: int = 5
    instrument_top_k: int = 3
    languages_top_k: int = 3
    edge_percentile: int = 66


class Settings(BaseModel):
    paths: PathsSettings
    download: DownloadSettings
    filter: FilterSettings
    mir: MirSettings = Field(default_factory=MirSettings)
    speech: SpeechSettings
    captions: CaptionsSettings
    embeddings: EmbeddingsSettings
    audio_extraction: AudioExtractionSettings = Field(
        default_factory=AudioExtractionSettings
    )
    search: SearchSettings
    validation: ValidationSettings
    storage: StorageSettings = Field(default_factory=StorageSettings)
    visualization: VisualizationSettings


class Secrets(BaseModel):
    database_url: str
    identity_db_url: str
    hiker_api_key: str
    huggingface_token: str
    embedder_remote_url: str = ""
    embedder_token: str = ""
    object_store_endpoint: str = ""
    object_store_access_key: str = ""
    object_store_secret_key: str = ""
    gemini_api_key: str | None = None


def load_runtime_config() -> tuple[Settings, Secrets]:
    with open(_CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)

    settings = Settings(
        paths=PathsSettings(**raw["paths"]),
        download=DownloadSettings(**raw["download"]),
        filter=FilterSettings(**raw.get("filter", {})),
        mir=MirSettings(**raw.get("mir", {})),
        speech=SpeechSettings(**raw["speech"]),
        captions=CaptionsSettings(**raw["captions"]),
        embeddings=EmbeddingsSettings(**raw["embeddings"]),
        audio_extraction=AudioExtractionSettings(**raw.get("audio_extraction", {})),
        search=SearchSettings(**raw.get("search", {})),
        validation=ValidationSettings(**raw["validation"]),
        storage=StorageSettings(**raw.get("storage", {})),
        visualization=VisualizationSettings(**raw["visualization"]),
    )

    gemini_enabled = settings.embeddings.gemini_enabled
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_enabled and not gemini_api_key:
        raise RuntimeError(
            "embeddings.gemini_enabled=true but GEMINI_API_KEY is not set"
        )

    secrets = Secrets(
        database_url=os.environ["DATABASE_URL"],
        identity_db_url=os.environ["IDENTITY_DB_URL"],
        hiker_api_key=os.environ["HIKER_API_KEY"],
        huggingface_token=os.environ["HUGGINGFACE_TOKEN"],
        embedder_remote_url=os.environ.get("EMBEDDER_REMOTE_URL", ""),
        embedder_token=os.environ.get("EMBEDDER_TOKEN", ""),
        object_store_endpoint=os.environ.get("OBJECT_STORE_ENDPOINT", ""),
        object_store_access_key=os.environ.get("OBJECT_STORE_ACCESS_KEY", ""),
        object_store_secret_key=os.environ.get("OBJECT_STORE_SECRET_KEY", ""),
        gemini_api_key=gemini_api_key if gemini_enabled else None,
    )

    # huggingface_hub reads HF_TOKEN, not HUGGINGFACE_TOKEN — propagate so gated
    # models (e.g. google/translategemma-4b-it) load without an interactive login.
    os.environ["HF_TOKEN"] = secrets.huggingface_token

    return settings, secrets
