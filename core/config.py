from __future__ import annotations

import os
import tomllib
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
_MUST_BE_POSITIVE = "must be > 0"


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
        "maest_patch_seconds",
    )
    @classmethod
    def _positive(cls, v: int | float) -> int | float:
        if v <= 0:
            raise ValueError(_MUST_BE_POSITIVE)
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


class LabelsSettings(BaseModel):
    model_path: str = "./models/Qwen3-VL-8B-Instruct"
    frame_count: int = 8
    max_new_tokens: int = 1200
    # Per-case output-token cap overrides. ``max_new_tokens`` is the global
    # default; a case listed here gets its own cap. The multimodal ``sandwich``
    # case produces the longest output and otherwise truncates at the global
    # cap (the JSON is cut mid-array, the bracket-balancer salvages it into a
    # wrong-key-set ``H2`` hard fail), so it carries a higher cap. Only the
    # text path (``run_text_batch``) honours this — the video case bakes
    # ``max_new_tokens`` into the engine at load time; that case is not
    # overridable here (and does not need to be). Drift is per-case isolated:
    # see ``clip_labels_config_payload`` in ``modules.labels.state``.
    clip_max_new_tokens_overrides: dict[str, int] = Field(default_factory=dict)
    max_attempts: int = 3
    parallelism: int = 1
    generation_seed: int = 0
    # GPU decode batch width for the clip-pass. Throughput knob only — excluded
    # from the clip-pass fingerprint (see ``_LABELS_CONFIG_FIELDS`` in
    # ``modules.labels.state``). Drives ``LabelsGenerator.run_many``.
    batch_size: int = 1
    # Clip-pass vLLM engine knobs (perf-only; excluded from the fingerprint).
    # Benchmarked throughput sweet spot on a 24 GB GPU (RTX 4090): max KV cache
    # for vLLM concurrency drives ~40% more clips/s than the conservative
    # 16384/0.90. 0.95 util OOMs; CUDA graphs (enforce_eager=False) trade KV for
    # no net gain. 6144 covers the worst clip/video sequence (~2.8k tokens) with
    # headroom — the real lever is KV (util), not max_model_len.
    clip_gpu_memory_utilization: float = 0.93
    clip_max_model_len: int = 6144
    clip_enforce_eager: bool = True

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
    cluster_model_path: str = "./models/Qwen3-30B-A3B-GPTQ-Int4"
    cluster_tag_max_chars: int = 28
    cluster_tag_max_words: int = 5
    cluster_summary_target_min: int = 120
    cluster_summary_target_max: int = 160
    cluster_summary_max_chars: int = 200
    # Stage-2 vLLM engine knobs (the cluster pass runs the 30B in-process via
    # vLLM's offline ``LLM`` with structured-output JSON decoding).
    # ``cluster_max_model_len`` must cover the largest cluster prompt
    # (bounded by ``cluster_sample_token_budget``) plus ``cluster_max_new_tokens``.
    cluster_gpu_memory_utilization: float = 0.90
    cluster_max_model_len: int = 20480
    cluster_enforce_eager: bool = True
    # Within-case label uniqueness: after labelling a case, the global naming
    # pass (modules.labels.cluster_naming) regenerates the FULL set of labels —
    # seeing every cluster at once so it can spread vocabulary — for this many
    # feedback rounds before the deterministic exact-uniqueness backstop.
    cluster_dedup_max_rounds: int = 2
    # Lead-in instructions for that global naming pass. Case-agnostic; the
    # roster, output contract and retry feedback are appended in code. Folded
    # into the cluster-pass config fingerprint per-case so an edit re-labels.
    cluster_naming_prompt: str = (
        "You assign short, distinct display names to a set of related clusters "
        "of Instagram Reels creators. You are given every cluster at once so "
        "you can make the names as mutually distinct as possible."
    )

    def max_new_tokens_for(self, case: str) -> int:
        """Output-token cap for ``case``: the per-case override if set, else the
        global ``max_new_tokens`` default."""
        return self.clip_max_new_tokens_overrides.get(case, self.max_new_tokens)

    @staticmethod
    def _promote_legacy(legacy: Any, sub_table: dict) -> tuple[dict, bool]:
        """Back-fill ``sub_table["video"]`` from a legacy key when unset.

        Returns the (possibly updated) sub-table and whether the legacy value
        was actually promoted (``False`` when ``video`` was already present).
        """
        used = legacy is not None and "video" not in sub_table
        if used:
            sub_table["video"] = legacy
        return sub_table, used

    @staticmethod
    def _warn_legacy_prompts(*, has_legacy: bool, both: bool) -> None:
        if not has_legacy:
            return
        note = (
            "labels.prompt / labels.cluster_prompt are deprecated; "
            "move them under [labels.case_prompts] / "
            "[labels.cluster_case_prompts]."
        )
        if both:
            note += (
                " Both legacy and new keys present — new keys win, legacy keys ignored."
            )
        warnings.warn(note, DeprecationWarning, stacklevel=2)

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
        case_prompts, used_clip = cls._promote_legacy(
            legacy_clip, dict(data.get("case_prompts") or {})
        )
        cluster_case_prompts, used_cluster = cls._promote_legacy(
            legacy_cluster, dict(data.get("cluster_case_prompts") or {})
        )

        both_clip = (
            legacy_clip is not None and "video" in case_prompts and not used_clip
        )
        both_cluster = (
            legacy_cluster is not None
            and "video" in cluster_case_prompts
            and not used_cluster
        )
        cls._warn_legacy_prompts(
            has_legacy=legacy_clip is not None or legacy_cluster is not None,
            both=both_clip or both_cluster,
        )

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
    # Cross-clip GPU coalescing for the local Qwen embedder. When >1, the
    # embedder runs up to ``embed_batch_size`` same-case clips in a single
    # padded forward. Throughput-only — not in ``case_config_identity``.
    embed_batch_size: int = 1

    @field_validator("embed_batch_size")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(_MUST_BE_POSITIVE)
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
    visualization: VisualizationSettings


class Secrets(BaseModel):
    database_url: str
    identity_db_url: str
    # Separate read-optimised store the offload script writes and the atlas
    # API reads. Defaults to a local SQLite file so dev/test never need to set
    # it; production points it at Postgres alongside DATABASE_URL.
    serving_database_url: str = "sqlite:///data/serving.db"
    hiker_api_key: str
    huggingface_token: str


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
        visualization=VisualizationSettings(**raw["visualization"]),
    )


def load_runtime_config() -> tuple[Settings, Secrets]:
    settings = _load_settings()

    serving_database_url = os.environ.get("SERVING_DATABASE_URL")
    secrets = Secrets(
        database_url=os.environ["DATABASE_URL"],
        identity_db_url=os.environ["IDENTITY_DB_URL"],
        **(
            {"serving_database_url": serving_database_url}
            if serving_database_url
            else {}
        ),
        hiker_api_key=os.environ["HIKER_API_KEY"],
        huggingface_token=os.environ["HUGGINGFACE_TOKEN"],
    )

    # huggingface_hub reads HF_TOKEN, not HUGGINGFACE_TOKEN — propagate so gated
    # models (e.g. google/translategemma-4b-it) load without an interactive login.
    os.environ["HF_TOKEN"] = secrets.huggingface_token

    return settings, secrets
