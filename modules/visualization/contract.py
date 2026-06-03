"""Pydantic mirror of the version-7 frontend JSON contract.

Mirrors the Zod schemas in ``frontend/src/data/schemas/*.ts`` field-for-field
(``extra="forbid"`` so any drift surfaces). These models validate every
payload the offload decomposes and the atlas API reconstructs, before bytes
are emitted — they never re-serialise (the exporter's serializer owns bytes),
they only assert shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.contract import SCHEMA_VERSION

# Literal needs a static value; assert it tracks the single-source constant so
# a schema bump trips here loudly rather than silently accepting stale payloads.
assert SCHEMA_VERSION == 7, "bump the _VERSION literal when SCHEMA_VERSION changes"
_VERSION = Literal[7]

# Cluster-label modality, surfaced on the lightweight main-detail payload so the
# viewer can place the tag skeleton (and decide whether to fetch a label at all)
# without downloading the heavy label block. None when the cluster has no label.
_LABEL_MODALITY = Literal["visual", "audio", "music", "multimodal", "textual"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ManifestRunModel(_Strict):
    id: str
    case: Literal["video", "sandwich", "auditory", "spoken", "textual"]
    label: str
    size: int = Field(ge=0)
    details_available: bool


class ManifestModel(_Strict):
    version: _VERSION
    default_run_id: str
    runs: list[ManifestRunModel] = Field(min_length=1)


class BoundsModel(_Strict):
    minX: float
    maxX: float
    minY: float
    maxY: float


# [id, x, y, cluster_id, has_detail, centrality]
UserTuple = tuple[int, float, float, int, bool, float]


class UsersFileModel(_Strict):
    version: _VERSION
    run_id: str
    bounds: BoundsModel
    users: list[UserTuple]


class ClusterEntryModel(_Strict):
    id: int
    label: str
    cx: float
    cy: float
    rx: float = Field(ge=0)
    ry: float = Field(ge=0)
    angle: float
    size: int = Field(ge=0)
    has_detail: bool


class ClustersFileModel(_Strict):
    version: _VERSION
    run_id: str
    clusters: list[ClusterEntryModel]


class AudioScoresModel(_Strict):
    approachability: float
    engagement: float
    danceability: float


class MoodSharesModel(_Strict):
    happy: float
    sad: float
    relaxed: float
    aggressive: float
    party: float


class TimbreSharesModel(_Strict):
    acoustic: float
    electronic: float
    instrumental: float
    female_voice: float
    bright: float
    tonal: float


class WeightedTagModel(_Strict):
    label: str
    weight: float = Field(ge=0)


class LangShareModel(_Strict):
    code: str
    share: float


class EllipseModel(_Strict):
    cx: float
    cy: float
    rx: float = Field(ge=0)
    ry: float = Field(ge=0)
    angle: float


class SpeechModel(_Strict):
    detected_share: float
    top_langs: list[LangShareModel]


class CaptionModel(_Strict):
    # Optional in Zod for back-compat, but the exporter always emits it.
    detected_share: float | None = None
    top_langs: list[LangShareModel]


class PostingModel(_Strict):
    median_plays: float
    median_clip_duration_s: float
    median_clips_per_week: float
    engagement_shape_ratio: float


class DistinctivenessEntryModel(_Strict):
    feature: str
    cohort_value: float
    baseline_mean: float
    baseline_std: float
    z: float


class NearestClusterModel(_Strict):
    cluster_id: int
    label: str
    distance: float


class RepertoireEntryModel(_Strict):
    tag: str
    description: str
    recurrence: Literal["dominant", "frequent", "occasional"]


class AestheticLogicEntryModel(_Strict):
    tag: str
    grounded_in: list[str]
    description: str


class CautiousBlockModel(_Strict):
    label: str
    description: str
    confidence: str


class InternalVariationModel(_Strict):
    variation: str
    description: str


class ClusterLabelModel(_Strict):
    label: str
    summary: str
    modality: Literal["visual", "audio", "music", "multimodal", "textual"]
    repertoire: list[RepertoireEntryModel]
    aesthetic_logic: list[AestheticLogicEntryModel]
    taste_signalling: CautiousBlockModel
    visibility_orientation: CautiousBlockModel
    internal_variations: list[InternalVariationModel]
    boundary_notes: str
    tool_tags: list[str]
    validation: Literal["ok", "warn"]
    warnings: list[str]


class SpatialClusterModel(_Strict):
    compactness: float
    nearest_clusters: list[NearestClusterModel]


class ClusterMainDetailModel(_Strict):
    """One cluster's main detail — everything except the heavy ``label`` block.

    Shipped eagerly in the per-run ``clusters-detail.json`` bundle. ``label``
    moved to its own per-cluster file (fetched on selection); ``label_modality``
    stays here so the viewer knows where the tags belong while they load.
    """

    cluster_id: int
    size: int = Field(ge=0)
    ellipse: EllipseModel
    audio: AudioScoresModel
    mood_shares: MoodSharesModel
    timbre_shares: TimbreSharesModel
    genre_top: list[WeightedTagModel]
    instrument_top: list[WeightedTagModel]
    speech: SpeechModel
    caption: CaptionModel
    posting: PostingModel
    follower_bucket: str
    activity_span_months: int
    distinctiveness: list[DistinctivenessEntryModel]
    spatial: SpatialClusterModel
    label_modality: _LABEL_MODALITY | None = None


class ClustersDetailBundleModel(_Strict):
    """``runs/<id>/clusters-detail.json`` — every cluster's main detail, one
    fetch per run."""

    version: _VERSION
    run_id: str
    clusters: list[ClusterMainDetailModel]


class ClusterLabelFileModel(_Strict):
    """``runs/<id>/clusters/<id>.label.json`` — the deferred tag block."""

    version: _VERSION
    cluster_id: int
    label: ClusterLabelModel


class NearestOtherClusterModel(_Strict):
    cluster_id: int
    label: str
    distance: float


class SpatialCreatorModel(_Strict):
    distance_from_centroid: float
    distance_from_centroid_percentile: float
    nearest_other_cluster: NearestOtherClusterModel | None


class ObservableTagModel(_Strict):
    tag: str
    evidence: str


class GroundedTagModel(_Strict):
    tag: str
    grounded_in: list[str]
    confidence: str


class ClipTagsModel(_Strict):
    observable: list[ObservableTagModel]
    aesthetic: list[GroundedTagModel]
    community: list[GroundedTagModel]


class ClipLabelEntryModel(_Strict):
    clip_id: int
    shortcode: str | None
    thumbnail_url: str | None
    sentence: str
    tags: ClipTagsModel
    validation: Literal["ok", "warn"]
    warnings: list[str]


class CreatorDetailModel(_Strict):
    version: _VERSION
    user_id: int = Field(ge=0)
    cluster_id: int
    x: float
    y: float
    n_clips: int = Field(ge=0)
    audio: AudioScoresModel
    mood_shares: MoodSharesModel
    timbre_shares: TimbreSharesModel
    genre_top: list[WeightedTagModel]
    instrument_top: list[WeightedTagModel]
    speech: SpeechModel
    caption: CaptionModel
    posting: PostingModel
    follower_bucket: str
    activity_span_months: int
    distinctiveness: list[DistinctivenessEntryModel]
    spatial: SpatialCreatorModel
    clips: list[ClipLabelEntryModel]
