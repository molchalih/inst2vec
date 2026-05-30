"""Serving-database schema (read-optimised, normalised).

The serving DB is a separate store: the offload script (``scripts/
offload_serving.py``) decomposes the version-6 frontend payloads into these
normalised tables, and the atlas API (``services/atlas_api``) reconstructs the
exact payloads from them. It carries only already-anonymised, already-shipped
fields — never PII.

Design: flat/bounded fields → columns; unbounded ordered arrays-of-objects →
child tables with an ``ord`` column preserving the exporter's emission order
(byte parity needs order preserved). Every table is keyed by ``run_id``
(= embedding_case) so multiple runs/cases coexist. ``grounded_in`` string
lists are stored as a single ``JSON`` column on their child row (R3) — the one
deliberate exception to full normalisation, to avoid a 4th child-table level.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import declarative_base

ServingBase = declarative_base()

_SERVING_RUN_ID = "serving_run.run_id"


# Classes use explicit ``Column`` objects (not ``Mapped[...]``) since the base
# comes from ``declarative_base()``; this keeps the schema readable and avoids
# typing friction on a base that ty already treats as Any.


# ── manifest + bulk ─────────────────────────────────────────────────────────


class ServingRun(ServingBase):
    """One row per run/case → a manifest entry plus manifest singletons."""

    __tablename__ = "serving_run"

    run_id = Column(String, primary_key=True)
    case = Column(String, nullable=False)
    label = Column(String, nullable=False)
    size = Column(Integer, nullable=False)
    details_available = Column(Boolean, nullable=False)
    # Manifest emission order (the exporter lists runs in `cases` order, not
    # alphabetically) — preserved so the reconstructed manifest is byte-identical.
    manifest_ord = Column(Integer, nullable=False)
    # Manifest singletons: exactly one run carries is_default=True and the
    # shared schema_version. The reconstruction reads them off any row.
    is_default = Column(Boolean, nullable=False, default=False)
    schema_version = Column(Integer, nullable=False)


class ServingRunBounds(ServingBase):
    __tablename__ = "serving_run_bounds"

    run_id = Column(String, ForeignKey(_SERVING_RUN_ID), primary_key=True)
    min_x = Column(Float, nullable=False)
    max_x = Column(Float, nullable=False)
    min_y = Column(Float, nullable=False)
    max_y = Column(Float, nullable=False)


class ServingUser(ServingBase):
    """The users.json 6-tuple: [id, x, y, cluster_id, has_detail, centrality]."""

    __tablename__ = "serving_user"

    run_id = Column(String, ForeignKey(_SERVING_RUN_ID), primary_key=True)
    user_id = Column(Integer, primary_key=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    cluster_id = Column(Integer, nullable=False)
    has_detail = Column(Boolean, nullable=False)
    centrality = Column(Float, nullable=False)


class ServingCluster(ServingBase):
    """A clusters.json entry: ellipse + label + has_detail flag."""

    __tablename__ = "serving_cluster"

    run_id = Column(String, ForeignKey(_SERVING_RUN_ID), primary_key=True)
    cluster_id = Column(Integer, primary_key=True)
    label = Column(String, nullable=False)
    cx = Column(Float, nullable=False)
    cy = Column(Float, nullable=False)
    rx = Column(Float, nullable=False)
    ry = Column(Float, nullable=False)
    angle = Column(Float, nullable=False)
    size = Column(Integer, nullable=False)
    has_detail = Column(Boolean, nullable=False)


# ── detail scalar blocks ────────────────────────────────────────────────────


class ServingUserDetail(ServingBase):
    """Per-(run,user) creator-detail scalar block."""

    __tablename__ = "serving_user_detail"

    run_id = Column(String, ForeignKey(_SERVING_RUN_ID), primary_key=True)
    user_id = Column(Integer, primary_key=True)
    # identity / position
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    n_clips = Column(Integer, nullable=False)
    cluster_id = Column(Integer, nullable=False)
    # audio (3)
    audio_approachability = Column(Float, nullable=False)
    audio_engagement = Column(Float, nullable=False)
    audio_danceability = Column(Float, nullable=False)
    # mood (5)
    mood_happy = Column(Float, nullable=False)
    mood_sad = Column(Float, nullable=False)
    mood_relaxed = Column(Float, nullable=False)
    mood_aggressive = Column(Float, nullable=False)
    mood_party = Column(Float, nullable=False)
    # timbre (6)
    timbre_acoustic = Column(Float, nullable=False)
    timbre_electronic = Column(Float, nullable=False)
    timbre_instrumental = Column(Float, nullable=False)
    timbre_female_voice = Column(Float, nullable=False)
    timbre_bright = Column(Float, nullable=False)
    timbre_tonal = Column(Float, nullable=False)
    # speech / caption shares (top_langs are child rows)
    speech_detected_share = Column(Float, nullable=False)
    caption_detected_share = Column(Float, nullable=True)
    # posting (4)
    posting_median_plays = Column(Integer, nullable=False)
    posting_median_clip_duration_s = Column(Float, nullable=False)
    posting_median_clips_per_week = Column(Float, nullable=False)
    posting_engagement_shape_ratio = Column(Float, nullable=False)
    # misc
    follower_bucket = Column(String, nullable=False)
    activity_span_months = Column(Integer, nullable=True)
    # spatial (creator)
    spatial_distance_from_centroid = Column(Float, nullable=False)
    # percentile is an integer rank (compute.centroid_percentile) — byte parity
    # needs it emitted as an int, not a float.
    spatial_distance_percentile = Column(Integer, nullable=False)
    # nearest_other_cluster — all-NULL ⇒ emit null
    spatial_nearest_cluster_id = Column(Integer, nullable=True)
    spatial_nearest_label = Column(String, nullable=True)
    spatial_nearest_distance = Column(Float, nullable=True)


class ServingClusterDetail(ServingBase):
    """Per-(run,cluster) cluster-detail scalar block."""

    __tablename__ = "serving_cluster_detail"

    run_id = Column(String, ForeignKey(_SERVING_RUN_ID), primary_key=True)
    cluster_id = Column(Integer, primary_key=True)
    size = Column(Integer, nullable=False)
    # ellipse (5)
    ellipse_cx = Column(Float, nullable=False)
    ellipse_cy = Column(Float, nullable=False)
    ellipse_rx = Column(Float, nullable=False)
    ellipse_ry = Column(Float, nullable=False)
    ellipse_angle = Column(Float, nullable=False)
    # audio (3)
    audio_approachability = Column(Float, nullable=False)
    audio_engagement = Column(Float, nullable=False)
    audio_danceability = Column(Float, nullable=False)
    # mood (5)
    mood_happy = Column(Float, nullable=False)
    mood_sad = Column(Float, nullable=False)
    mood_relaxed = Column(Float, nullable=False)
    mood_aggressive = Column(Float, nullable=False)
    mood_party = Column(Float, nullable=False)
    # timbre (6)
    timbre_acoustic = Column(Float, nullable=False)
    timbre_electronic = Column(Float, nullable=False)
    timbre_instrumental = Column(Float, nullable=False)
    timbre_female_voice = Column(Float, nullable=False)
    timbre_bright = Column(Float, nullable=False)
    timbre_tonal = Column(Float, nullable=False)
    # speech / caption shares
    speech_detected_share = Column(Float, nullable=False)
    caption_detected_share = Column(Float, nullable=True)
    # posting (4)
    posting_median_plays = Column(Integer, nullable=False)
    posting_median_clip_duration_s = Column(Float, nullable=False)
    posting_median_clips_per_week = Column(Float, nullable=False)
    posting_engagement_shape_ratio = Column(Float, nullable=False)
    # misc
    follower_bucket = Column(String, nullable=False)
    activity_span_months = Column(Integer, nullable=True)
    # spatial (cluster)
    spatial_compactness = Column(Float, nullable=False)


# ── unbounded child tables shared by both detail kinds ──────────────────────


class ServingWeightedTag(ServingBase):
    """genre_top / instrument_top entries for user|cluster details."""

    __tablename__ = "serving_weighted_tag"

    run_id = Column(String, primary_key=True)
    owner_kind = Column(String, primary_key=True)  # "user" | "cluster"
    owner_id = Column(Integer, primary_key=True)
    field = Column(String, primary_key=True)  # "genre" | "instrument"
    ord = Column(Integer, primary_key=True)
    label = Column(String, nullable=False)
    weight = Column(Float, nullable=False)


class ServingLangShare(ServingBase):
    """speech.top_langs / caption.top_langs for user|cluster details."""

    __tablename__ = "serving_lang_share"

    run_id = Column(String, primary_key=True)
    owner_kind = Column(String, primary_key=True)  # "user" | "cluster"
    owner_id = Column(Integer, primary_key=True)
    block = Column(String, primary_key=True)  # "speech" | "caption"
    ord = Column(Integer, primary_key=True)
    code = Column(String, nullable=False)
    share = Column(Float, nullable=False)


class ServingDistinctiveness(ServingBase):
    """distinctiveness[] for user|cluster details."""

    __tablename__ = "serving_distinctiveness"

    run_id = Column(String, primary_key=True)
    owner_kind = Column(String, primary_key=True)
    owner_id = Column(Integer, primary_key=True)
    ord = Column(Integer, primary_key=True)
    feature = Column(String, nullable=False)
    cohort_value = Column(Float, nullable=False)
    baseline_mean = Column(Float, nullable=False)
    baseline_std = Column(Float, nullable=False)
    z = Column(Float, nullable=False)


class ServingClusterNearest(ServingBase):
    """spatial.nearest_clusters[] (cluster detail only)."""

    __tablename__ = "serving_cluster_nearest"

    run_id = Column(String, primary_key=True)
    cluster_id = Column(Integer, primary_key=True)
    ord = Column(Integer, primary_key=True)
    nearest_cluster_id = Column(Integer, nullable=False)
    label = Column(String, nullable=False)
    distance = Column(Float, nullable=False)


# ── cluster label block (optional, row-presence gated) ──────────────────────


class ServingClusterLabel(ServingBase):
    """The optional cluster ``label`` block scalars; row presence ⇔ block."""

    __tablename__ = "serving_cluster_label"

    run_id = Column(String, primary_key=True)
    cluster_id = Column(Integer, primary_key=True)
    label = Column(String, nullable=False)
    summary = Column(String, nullable=False)
    modality = Column(String, nullable=False)
    taste_signalling_label = Column(String, nullable=False)
    taste_signalling_description = Column(String, nullable=False)
    taste_signalling_confidence = Column(String, nullable=False)
    visibility_orientation_label = Column(String, nullable=False)
    visibility_orientation_description = Column(String, nullable=False)
    visibility_orientation_confidence = Column(String, nullable=False)
    boundary_notes = Column(String, nullable=False)
    validation = Column(String, nullable=False)


class ServingClusterLabelRepertoire(ServingBase):
    __tablename__ = "serving_cluster_label_repertoire"

    run_id = Column(String, primary_key=True)
    cluster_id = Column(Integer, primary_key=True)
    ord = Column(Integer, primary_key=True)
    tag = Column(String, nullable=False)
    description = Column(String, nullable=False)
    recurrence = Column(String, nullable=False)


class ServingClusterLabelAesthetic(ServingBase):
    __tablename__ = "serving_cluster_label_aesthetic"

    run_id = Column(String, primary_key=True)
    cluster_id = Column(Integer, primary_key=True)
    ord = Column(Integer, primary_key=True)
    tag = Column(String, nullable=False)
    grounded_in = Column(JSON, nullable=False)  # string[] (R3)
    description = Column(String, nullable=False)


class ServingClusterLabelVariations(ServingBase):
    __tablename__ = "serving_cluster_label_variations"

    run_id = Column(String, primary_key=True)
    cluster_id = Column(Integer, primary_key=True)
    ord = Column(Integer, primary_key=True)
    variation = Column(String, nullable=False)
    description = Column(String, nullable=False)


class ServingClusterLabelTooltag(ServingBase):
    """label.tool_tags[] + label.warnings[] string lists."""

    __tablename__ = "serving_cluster_label_tooltag"

    run_id = Column(String, primary_key=True)
    cluster_id = Column(Integer, primary_key=True)
    kind = Column(String, primary_key=True)  # "tool_tag" | "warning"
    ord = Column(Integer, primary_key=True)
    value = Column(String, nullable=False)


# ── creator clips[] (+ nested tags / warnings) ──────────────────────────────


class ServingUserClip(ServingBase):
    __tablename__ = "serving_user_clip"

    run_id = Column(String, primary_key=True)
    user_id = Column(Integer, primary_key=True)
    ord = Column(Integer, primary_key=True)
    clip_id = Column(Integer, nullable=False)
    shortcode = Column(String, nullable=True)
    thumbnail_url = Column(String, nullable=True)
    sentence = Column(String, nullable=False)
    validation = Column(String, nullable=False)


class ServingUserClipTag(ServingBase):
    """Per-clip observable / aesthetic / community tags.

    Observable tags carry ``tag`` + ``evidence``; grounded (aesthetic /
    community) tags carry ``tag`` + ``grounded_in`` (JSON string[]) +
    ``confidence``. Unused columns are NULL per ``kind``.
    """

    __tablename__ = "serving_user_clip_tag"

    run_id = Column(String, primary_key=True)
    user_id = Column(Integer, primary_key=True)
    clip_ord = Column(Integer, primary_key=True)
    kind = Column(String, primary_key=True)  # observable|aesthetic|community
    ord = Column(Integer, primary_key=True)
    tag = Column(String, nullable=False)
    evidence = Column(String, nullable=True)  # observable only
    grounded_in = Column(JSON, nullable=True)  # grounded only (R3)
    confidence = Column(String, nullable=True)  # grounded only


class ServingUserClipWarning(ServingBase):
    __tablename__ = "serving_user_clip_warning"

    run_id = Column(String, primary_key=True)
    user_id = Column(Integer, primary_key=True)
    clip_ord = Column(Integer, primary_key=True)
    ord = Column(Integer, primary_key=True)
    value = Column(String, nullable=False)
