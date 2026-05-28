from __future__ import annotations

import os
import tomllib
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

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


class ParseSettings(BaseModel):
    max_attempts: int = 3
    retry_delay: int = 30
    retry_jitter: int = 5
    concurrency: int = 5


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
    maest_onnx_checkpoint: str = "discogs-maest-30s-pw-519l-1.onnx"
    effnet_onnx_checkpoint: str = "discogs-effnet-bsdynamic-1.onnx"

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


class LabelsSettings(BaseModel):
    model_path: str = "./models/Qwen3-VL-8B-Instruct"
    frame_count: int = 8
    max_new_tokens: int = 1200
    max_attempts: int = 3
    parallelism: int = 1
    generation_seed: int = 0
    # GPU decode batch width for the clip-pass. Throughput knob only — excluded
    # from the clip-pass fingerprint (see ``_LABELS_CONFIG_FIELDS`` in
    # ``modules.labels.state``). Drives ``LabelsGenerator.run_many``.
    batch_size: int = 1

    min_tags_per_kind: int = 3
    max_tags_per_kind: int = 10
    min_tag_chars: int = 3
    max_tag_chars: int = 60
    min_sentence_chars: int = 20
    max_sentence_chars: int = 240

    # Per-case prompts. ``case_prompts[<case>]`` is the stage-1 prompt body
    # the labels pass renders against ``modules.labels.cases.REGISTRY[case]``.
    # Mirrors the per-case shape of ``modules.embeddings.cases.CASE_REGISTRY``.
    # The legacy flat ``labels.prompt`` / ``labels.cluster_prompt`` keys are
    # accepted with a ``DeprecationWarning`` (see ``_promote_legacy_prompts``)
    # and back-fill into ``case_prompts["video"]`` only — non-video cases
    # raise a clear ``ValueError`` at ``modules.labels.prompts.prompt_for``
    # lookup time when their entry is missing.
    case_prompts: dict[str, str] = Field(default_factory=dict)
    cluster_case_prompts: dict[str, str] = Field(default_factory=dict)

    # Stage 2 (per-cluster pass) knobs.
    cluster_max_new_tokens: int = 1400
    cluster_sample_token_budget: int = 7500
    cluster_max_clips_per_cluster: int = 60
    cluster_max_clips_per_user: int = 2
    cluster_min_tags: int = 3
    cluster_max_tags: int = 12
    cluster_min_sentence_chars: int = 20
    cluster_max_sentence_chars: int = 320
    cluster_max_attempts: int = 3

    @model_validator(mode="before")
    @classmethod
    def _promote_legacy_prompts(cls, data: Any) -> Any:
        """Promote legacy flat ``prompt`` / ``cluster_prompt`` keys into the
        per-case sub-tables (SPEC §5.6).

        Fires a single ``DeprecationWarning`` per ``LabelsSettings`` load
        when either legacy key is present. When both old and new keys are
        supplied, the new sub-tables win and the warning still fires so the
        operator is told to drop the dead flat key.
        """
        if not isinstance(data, dict):
            return data
        legacy_clip = data.pop("prompt", None)
        legacy_cluster = data.pop("cluster_prompt", None)
        case_prompts = dict(data.get("case_prompts") or {})
        cluster_case_prompts = dict(data.get("cluster_case_prompts") or {})

        used_legacy_clip = legacy_clip is not None and "video" not in case_prompts
        used_legacy_cluster = (
            legacy_cluster is not None and "video" not in cluster_case_prompts
        )
        if used_legacy_clip:
            case_prompts["video"] = legacy_clip
        if used_legacy_cluster:
            cluster_case_prompts["video"] = legacy_cluster

        both_clip = (
            legacy_clip is not None and "video" in case_prompts and not used_legacy_clip
        )
        both_cluster = (
            legacy_cluster is not None
            and "video" in cluster_case_prompts
            and not used_legacy_cluster
        )
        if legacy_clip is not None or legacy_cluster is not None:
            note = (
                "labels.prompt / labels.cluster_prompt are deprecated; "
                "move them under [labels.case_prompts] / "
                "[labels.cluster_case_prompts]."
            )
            if both_clip or both_cluster:
                note += (
                    " Both legacy and new keys present — new keys win, "
                    "legacy keys ignored."
                )
            warnings.warn(note, DeprecationWarning, stacklevel=2)

        if case_prompts:
            data["case_prompts"] = case_prompts
        if cluster_case_prompts:
            data["cluster_case_prompts"] = cluster_case_prompts
        return data


class SpeechSettings(BaseModel):
    whisper_model: str
    commit_every: int
    translate_model: str
    translate_target_lang: str
    translation_max_chars: int
    translate_max_new_tokens: int
    # GPU decode batch width for translation. Throughput-only — does not change
    # per-row outputs, so it is excluded from the speech config fingerprint.
    translate_batch_size: int = 16
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
    # GPU decode batch width for translation. Throughput-only — does not change
    # per-row outputs, so it is excluded from the captions config fingerprint.
    translate_batch_size: int = 16


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
    inflight: int = 1
    # ── distributed coordinator / worker ──
    coordinator_bind_host: str = "0.0.0.0"
    coordinator_bind_port: int = 8765
    lease_ttl_s: int = 600
    max_attempts: int = 3
    worker_request_timeout_s: int = 120
    worker_max_retries: int = 3
    pod_connect_timeout_s: int = 600
    # Seconds the coordinator keeps serving after the queue drains so pods
    # polling /lease observe HTTP 410 instead of a connection error on shutdown.
    pod_drain_grace_s: float = 10.0
    # Per-iteration timeout for the orchestrator drain loop's queue.get().
    # Production default is fine; tests override to ~0.01 to avoid stacking
    # half-second blocks when the fake-provider work is sub-millisecond.
    drain_poll_s: float = 0.5
    # Seconds a pod tolerates an unreachable coordinator before exiting cleanly
    # (crash backstop so a dead orchestrator doesn't leave GPUs billing).
    pod_idle_ttl_s: int = 300
    # ── Gemini Embedding 2 case ──
    gemini_enabled: bool = False
    gemini_model: str = "gemini-embedding-2-preview"
    gemini_output_dim: int = 3072
    gemini_max_video_seconds: int = 120
    gemini_max_audio_seconds: int = 80
    gemini_request_timeout_s: int = 60
    gemini_max_retries: int = 5

    @field_validator(
        "inflight",
        "coordinator_bind_port",
        "lease_ttl_s",
        "max_attempts",
        "worker_request_timeout_s",
        "worker_max_retries",
        "pod_connect_timeout_s",
        "pod_idle_ttl_s",
    )
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
    # Cap the largest cluster at this fraction of the analysis users
    # (HDBSCAN max_cluster_size = round(frac * n)). Only HDBSCAN's eom
    # selection honors it; leaf ignores it. 0.0 disables the cap.
    hdbscan_max_cluster_frac: float = 0.0
    # Per-case embedding preprocessing applied in load_user_matrix before
    # UMAP: "none" | "center" (subtract column mean) | "standardize"
    # (z-score). Cases absent from the map default to "none".
    embedding_preprocess: dict[str, str] = {}
    random_state: int = 42


class ValidationSettings(BaseModel):
    plateau_drop_threshold: float
    max_noise_ratio: float
    min_clusters: int
    max_clusters: int
    # Reject runs whose largest cluster exceeds this fraction of assigned
    # (non-noise) users. 1.0 disables the guard.
    max_dominance: float = 1.0


class StorageSettings(BaseModel):
    backend: str = "s3"
    bucket: str = ""
    prefix: str = "videos/"
    # SigV4 region. Required for RunPod's S3 endpoint (= the datacenter id, e.g.
    # "EU-RO-1"), else SignatureDoesNotMatch. Empty for AWS/R2 default.
    region: str = ""
    # Thread-pool width for the upload stage's verify+upload scan. Every rerun
    # HEADs every selected clip to keep the bucket authoritative; these are tiny
    # network calls, so a wide pool keeps reruns fast. Independent of download
    # concurrency (which throttles HikerAPI/CDN pulls, not S3 HEADs).
    verify_concurrency: int = 64
    # How many actual video uploads (store.put) may run at once. Held well below
    # verify_concurrency so a cold/mismatched bucket cannot turn the wide HEAD
    # pool into a verify-width burst of multi-MB PUTs that throttles the endpoint.
    upload_concurrency: int = 8

    @field_validator("verify_concurrency", "upload_concurrency")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class RunpodSettings(BaseModel):
    image: str = ""
    gpu_type_id: str = ""
    data_center_id: str = ""
    network_volume_id: str = ""
    volume_mount_path: str = "/runpod-volume"
    container_disk_in_gb: int = 20
    pod_video_root: str = "/runpod-volume/videos"
    pod_model_path: str = "/runpod-volume/models/Qwen3-VL-Embedding-8B"
    reconcile_path: str = ".runpod_fleet.json"
    # RunPod template id to launch from when the image is in a private registry
    # (the template holds the pull credential). Empty -> deploy from `image`.
    template_id: str = ""
    # GPU selection. Pin gpu_type_id to force one type; leave it empty to
    # auto-fetch in-stock types in the volume's data_center_id that meet the
    # VRAM/RAM floors and sit under the price cap, tried cheapest-first.
    gpu_max_price_hr: float = 0.80
    gpu_min_vram_gb: int = 24
    gpu_min_ram_gb: int = 30
    # Background top-up: re-fetch availability and deploy any shortfall every
    # this many seconds, for the whole embedding stage, until the pod count is met.
    gpu_poll_interval_s: float = 30.0


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
    parse: ParseSettings = Field(default_factory=ParseSettings)
    filter: FilterSettings
    mir: MirSettings = Field(default_factory=MirSettings)
    labels: LabelsSettings
    speech: SpeechSettings
    captions: CaptionsSettings
    embeddings: EmbeddingsSettings
    audio_extraction: AudioExtractionSettings = Field(
        default_factory=AudioExtractionSettings
    )
    search: SearchSettings
    validation: ValidationSettings
    storage: StorageSettings = Field(default_factory=StorageSettings)
    runpod: RunpodSettings = Field(default_factory=RunpodSettings)
    visualization: VisualizationSettings


class Secrets(BaseModel):
    database_url: str
    identity_db_url: str
    hiker_api_key: str
    huggingface_token: str
    embedder_token: str = ""
    object_store_endpoint: str = ""
    object_store_access_key: str = ""
    object_store_secret_key: str = ""
    gemini_api_key: str | None = None
    runpod_api_key: str = ""
    coordinator_public_host: str = ""


def _load_settings() -> Settings:
    with open(_CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)

    return Settings(
        paths=PathsSettings(**raw["paths"]),
        download=DownloadSettings(**raw["download"]),
        parse=ParseSettings(**raw.get("parse", {})),
        filter=FilterSettings(**raw.get("filter", {})),
        mir=MirSettings(**raw.get("mir", {})),
        labels=LabelsSettings(**raw["labels"]),
        speech=SpeechSettings(**raw["speech"]),
        captions=CaptionsSettings(**raw["captions"]),
        embeddings=EmbeddingsSettings(**raw["embeddings"]),
        audio_extraction=AudioExtractionSettings(**raw.get("audio_extraction", {})),
        search=SearchSettings(**raw.get("search", {})),
        validation=ValidationSettings(**raw["validation"]),
        storage=StorageSettings(**raw.get("storage", {})),
        runpod=RunpodSettings(**raw.get("runpod", {})),
        visualization=VisualizationSettings(**raw["visualization"]),
    )


def load_pod_config() -> Settings:
    """Settings for an embedding pod — no pipeline secrets required.

    A pod only embeds: it leases jobs over HTTP, runs the local Qwen model
    on its own GPU, and reports vectors back. It never touches the DB or
    ingest APIs, so it must not require DATABASE_URL / IDENTITY_DB_URL /
    HIKER_API_KEY the way load_runtime_config does. Embedding tunables come
    from config.toml (shipped in the image) so they match the orchestrator;
    MODEL_PATH overrides the model location for the pod's mounted volume.
    """
    settings = _load_settings()
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        settings.paths.model_path = model_path
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    return settings


def _apply_deployment_env_overrides(settings: Settings) -> None:
    """Override deployment-specific RunPod/storage fields from the environment.

    These values are account/run-specific (volume id, datacenter, GPU pick), so
    they belong in .env rather than the tracked config.toml. Each var is
    optional and only overrides when set; bucket and region default to the
    network volume id and its datacenter so .env need only state them once.
    """
    rp = settings.runpod
    for attr, env in (
        ("image", "RUNPOD_IMAGE"),
        ("template_id", "RUNPOD_TEMPLATE_ID"),
        ("gpu_type_id", "RUNPOD_GPU_TYPE_ID"),
        ("data_center_id", "RUNPOD_DATA_CENTER_ID"),
        ("network_volume_id", "RUNPOD_NETWORK_VOLUME_ID"),
    ):
        val = os.environ.get(env)
        if val:
            setattr(rp, attr, val)
    if os.environ.get("RUNPOD_GPU_MAX_PRICE_HR"):
        rp.gpu_max_price_hr = float(os.environ["RUNPOD_GPU_MAX_PRICE_HR"])
    if os.environ.get("RUNPOD_GPU_MIN_VRAM_GB"):
        rp.gpu_min_vram_gb = int(os.environ["RUNPOD_GPU_MIN_VRAM_GB"])
    if os.environ.get("RUNPOD_GPU_MIN_RAM_GB"):
        rp.gpu_min_ram_gb = int(os.environ["RUNPOD_GPU_MIN_RAM_GB"])

    st = settings.storage
    if os.environ.get("STORAGE_BUCKET"):
        st.bucket = os.environ["STORAGE_BUCKET"]
    if os.environ.get("STORAGE_REGION"):
        st.region = os.environ["STORAGE_REGION"]
    # The bucket *is* the network volume and the region *is* its datacenter.
    if not st.bucket and rp.network_volume_id:
        st.bucket = rp.network_volume_id
    if not st.region and rp.data_center_id:
        st.region = rp.data_center_id


def load_runpod_config() -> tuple[Settings, str]:
    """Settings + RUNPOD_API_KEY for RunPod-only helpers (e.g. GPU listing).

    Loads non-secret settings, applies the deployment env overrides (so the
    network volume / datacenter pulled from .env are present), and returns the
    RunPod API key — without requiring pipeline secrets (DB / Hiker /
    HuggingFace) or running Gemini validation the way load_runtime_config does.
    Lets an operator list GPU availability from a minimal RunPod-only env.
    """
    settings = _load_settings()
    _apply_deployment_env_overrides(settings)
    return settings, os.environ.get("RUNPOD_API_KEY", "")


def load_runtime_config() -> tuple[Settings, Secrets]:
    settings = _load_settings()
    _apply_deployment_env_overrides(settings)

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
        embedder_token=os.environ.get("EMBEDDER_TOKEN", ""),
        object_store_endpoint=os.environ.get("OBJECT_STORE_ENDPOINT", ""),
        object_store_access_key=os.environ.get("OBJECT_STORE_ACCESS_KEY", ""),
        object_store_secret_key=os.environ.get("OBJECT_STORE_SECRET_KEY", ""),
        gemini_api_key=gemini_api_key if gemini_enabled else None,
        runpod_api_key=os.environ.get("RUNPOD_API_KEY", ""),
        coordinator_public_host=os.environ.get("COORDINATOR_PUBLIC_HOST", ""),
    )

    # huggingface_hub reads HF_TOKEN, not HUGGINGFACE_TOKEN — propagate so gated
    # models (e.g. google/translategemma-4b-it) load without an interactive login.
    os.environ["HF_TOKEN"] = secrets.huggingface_token

    return settings, secrets
