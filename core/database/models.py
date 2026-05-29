"""Main-DB ORM: Base + model classes. No engine handles, no predicates."""

from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    following_count: Mapped[int | None] = mapped_column(Integer)
    follower_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parse_status: Mapped[str | None] = mapped_column(String, nullable=True)
    is_low_plays_median: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_not_enough_preprocessed: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    is_not_enough_eligible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_eligible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_selected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    clips: Mapped[list["Clip"]] = relationship("Clip", back_populates="user")  # type: ignore[assignment]
    embeddings: Mapped[list["UserEmbedding"]] = relationship(  # type: ignore[assignment]
        "UserEmbedding", back_populates="user"
    )
    clusters: Mapped[list["UserCluster"]] = relationship(  # type: ignore[assignment]
        "UserCluster", back_populates="user"
    )


class UserStats(Base):
    __tablename__ = "user_stats"

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), primary_key=True
    )
    n_clips: Mapped[int | None] = mapped_column(Integer, nullable=True)
    median_plays: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_plays: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_plays: Mapped[int | None] = mapped_column(Integer, nullable=True)
    min_plays: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mean_log_plays: Mapped[float | None] = mapped_column(Float, nullable=True)
    median_log_plays: Mapped[float | None] = mapped_column(Float, nullable=True)
    log_plays_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    log_plays_mad: Mapped[float | None] = mapped_column(Float, nullable=True)
    top_to_median_plays_ratio: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    share_of_plays_from_top_clip: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    median_video_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_video_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    oldest_clip_taken_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    newest_clip_taken_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    clip_time_span_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    approx_clips_per_week: Mapped[float | None] = mapped_column(Float, nullable=True)
    computed_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=False
    )
    thumbnail_url: Mapped[str | None] = mapped_column(String)
    video_url: Mapped[str | None] = mapped_column(String)
    caption_text: Mapped[str | None] = mapped_column(Text)
    caption_clean: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption_language: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption_translation: Mapped[str | None] = mapped_column(Text)
    comment_count: Mapped[int | None] = mapped_column(Integer)
    reshare_count: Mapped[int | None] = mapped_column(Integer)
    like_count: Mapped[int | None] = mapped_column(Integer)
    play_count: Mapped[int | None] = mapped_column(Integer)
    video_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    taken_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    speech_transcription: Mapped[str | None] = mapped_column(Text)
    speech_language: Mapped[str | None] = mapped_column(String, nullable=True)
    speech_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    speech_avg_logprob: Mapped[float | None] = mapped_column(Float, nullable=True)
    speech_compression_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_speech_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    speech_translation: Mapped[str | None] = mapped_column(Text)
    is_garbage: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_too_short: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_too_long: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_too_old: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_low_percentile: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_high_percentile: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_preprocessed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_eligible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_selected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_downloaded: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_uploaded: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="clips")  # type: ignore[assignment]
    embeddings: Mapped[list["ClipEmbedding"]] = relationship(  # type: ignore[assignment]
        "ClipEmbedding", back_populates="clip"
    )


class ClipFilterScratch(Base):
    __tablename__ = "clip_filter_scratch"

    clip_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("clips.id"), primary_key=True
    )
    log_plays: Mapped[float | None] = mapped_column(Float, nullable=True)
    creator_relative_robust_z: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    is_creator_low_outlier: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


class AudioMIR(Base):
    __tablename__ = "audio_mir"
    __table_args__ = (UniqueConstraint("clip_id", name="uq_audio_mir_clip"),)

    clip_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("clips.id"), primary_key=True
    )

    is_mir_extracted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_music_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    mir_error: Mapped[str | None] = mapped_column(
        SAEnum(
            "maest",
            "effnet",
            "audio_load",
            "no_audio_file",
            name="audio_mir_error",
            validate_strings=True,
        ),
        nullable=True,
    )

    approachability: Mapped[float | None] = mapped_column(Float, nullable=True)
    engagement: Mapped[float | None] = mapped_column(Float, nullable=True)
    danceability: Mapped[float | None] = mapped_column(Float, nullable=True)

    is_aggressive: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_happy: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_party: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_relaxed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_sad: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_acoustic: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_electronic: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_instrumental: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_female_voice: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_bright_timbre: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_tonal: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    genre_labels: Mapped[str | None] = mapped_column(String, nullable=True)
    genre_scores: Mapped[str | None] = mapped_column(String, nullable=True)
    moodtheme_labels: Mapped[str | None] = mapped_column(String, nullable=True)
    moodtheme_scores: Mapped[str | None] = mapped_column(String, nullable=True)
    instrument_labels: Mapped[str | None] = mapped_column(String, nullable=True)
    instrument_scores: Mapped[str | None] = mapped_column(String, nullable=True)

    audio_duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    clip: Mapped["Clip"] = relationship("Clip")


class ClipEmbedding(Base):
    __tablename__ = "clip_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "clip_id", "embedding_case", name="uq_clip_embeddings_clip_case"
        ),
    )

    clip_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("clips.id"), primary_key=True
    )
    embedding_case: Mapped[str] = mapped_column(String, primary_key=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    source_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    clip: Mapped["Clip"] = relationship("Clip", back_populates="embeddings")  # type: ignore[assignment]


class UserEmbedding(Base):
    __tablename__ = "user_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "embedding_case", name="uq_user_embeddings_user_case"
        ),
    )

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), primary_key=True
    )
    embedding_case: Mapped[str] = mapped_column(String, primary_key=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    source_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship("User", back_populates="embeddings")  # type: ignore[assignment]


class UserCluster(Base):
    __tablename__ = "user_clusters"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "embedding_case", name="uq_user_clusters_user_case"
        ),
    )

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), primary_key=True
    )
    embedding_case: Mapped[str] = mapped_column(String, primary_key=True)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    umap_x: Mapped[float] = mapped_column(Float, nullable=False)
    umap_y: Mapped[float] = mapped_column(Float, nullable=False)
    # HDBSCAN soft membership probability in [0, 1]: 1 = core, 0 = barely in /
    # noise. Drives the size-by-centrality encoding in plots and the frontend.
    centrality: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    user: Mapped["User"] = relationship("User", back_populates="clusters")  # type: ignore[assignment]


class StageState(Base):
    __tablename__ = "stage_state"

    stage_name: Mapped[str] = mapped_column(String, primary_key=True)
    scope_key: Mapped[str] = mapped_column(String, primary_key=True)
    data_hash: Mapped[str] = mapped_column(String, nullable=False)
    config_hash: Mapped[str] = mapped_column(String, nullable=False)
    dependency_hash: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class ClusterRun(Base):
    __tablename__ = "cluster_runs"
    __table_args__ = (
        UniqueConstraint(
            "embedding_case",
            "umap_n_components",
            "umap_n_neighbors",
            "umap_min_dist",
            "umap_metric",
            "umap2d_n_neighbors",
            "umap2d_min_dist",
            "umap2d_metric",
            "hdbscan_min_cluster_size",
            "hdbscan_min_samples",
            "hdbscan_cluster_selection_method",
            "hdbscan_metric",
            "random_state",
            name="uq_cluster_runs_params",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    embedding_case: Mapped[str] = mapped_column(String, nullable=False)
    umap_n_components: Mapped[int] = mapped_column(Integer, nullable=False)
    umap_n_neighbors: Mapped[int] = mapped_column(Integer, nullable=False)
    umap_min_dist: Mapped[float] = mapped_column(Float, nullable=False)
    umap_metric: Mapped[str] = mapped_column(String, nullable=False)
    umap2d_n_neighbors: Mapped[int] = mapped_column(Integer, nullable=False)
    umap2d_min_dist: Mapped[float] = mapped_column(Float, nullable=False)
    umap2d_metric: Mapped[str] = mapped_column(String, nullable=False)
    hdbscan_min_cluster_size: Mapped[int] = mapped_column(Integer, nullable=False)
    hdbscan_min_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hdbscan_cluster_selection_method: Mapped[str] = mapped_column(
        String, nullable=False
    )
    hdbscan_metric: Mapped[str] = mapped_column(String, nullable=False)
    random_state: Mapped[int] = mapped_column(Integer, nullable=False)
    n_clusters: Mapped[int] = mapped_column(Integer, nullable=False)
    noise_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    min_size: Mapped[int] = mapped_column(Integer, nullable=False)
    median_size: Mapped[int] = mapped_column(Integer, nullable=False)
    max_size: Mapped[int] = mapped_column(Integer, nullable=False)
    passes_validation: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    dbcv: Mapped[float | None] = mapped_column(Float, nullable=True)
    silhouette: Mapped[float | None] = mapped_column(Float, nullable=True)
    param_plateau_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Visualization(Base):
    __tablename__ = "visualizations"

    embedding_case: Mapped[str] = mapped_column(String, primary_key=True)
    label: Mapped[str] = mapped_column(String, nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    source_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class VisualizationUser(Base):
    __tablename__ = "visualization_users"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "embedding_case", name="uq_visualization_users_user_case"
        ),
    )

    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id"), primary_key=True
    )
    embedding_case: Mapped[str] = mapped_column(String, primary_key=True)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    centrality: Mapped[float | None] = mapped_column(Float, nullable=True)


class VisualizationCluster(Base):
    __tablename__ = "visualization_clusters"

    embedding_case: Mapped[str] = mapped_column(String, primary_key=True)
    cluster_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cx: Mapped[float] = mapped_column(Float, nullable=False)
    cy: Mapped[float] = mapped_column(Float, nullable=False)
    rx: Mapped[float] = mapped_column(Float, nullable=False)
    ry: Mapped[float] = mapped_column(Float, nullable=False)
    angle: Mapped[float] = mapped_column(Float, nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String, nullable=False)


class ClipLabel(Base):
    __tablename__ = "clip_labels"
    __table_args__ = (
        UniqueConstraint("clip_id", "label_case", name="uq_clip_labels_clip_case"),
    )

    clip_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("clips.id"), primary_key=True
    )
    # Per-case discriminator mirroring ``ClipEmbedding.embedding_case`` so a
    # single clip can carry one stage-1 label row per modality (video / audio
    # / sandwich / maest / gemini). The ``server_default="video"`` is the
    # in-place backfill mechanism for sqlite (no Alembic); pre-existing rows
    # come back as ``label_case="video"`` on the next ``init_db`` run.
    label_case: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False, server_default="video"
    )
    status: Mapped[str] = mapped_column(String, nullable=False)
    validation: Mapped[str | None] = mapped_column(String, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    warnings: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    # Seed used by ``torch.manual_seed`` for the row's current state
    # (the call that produced the success payload, or the latest failed
    # attempt). Per-attempt seed variation gives validation hard fails a
    # real chance at recovery — same prompt with a different seed
    # produces different greedy paths.
    generation_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Stable digest of the case's input text (audio / sandwich / maest).
    # ``None`` for the video case — frames are fingerprinted by file-stat
    # upstream, not by a per-row source string. Mirrors
    # ``ClipEmbedding.source_hash``.
    source_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    clip: Mapped["Clip"] = relationship("Clip")


class ClusterLabel(Base):
    __tablename__ = "cluster_labels"

    embedding_case: Mapped[str] = mapped_column(String, primary_key=True)
    cluster_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    validation: Mapped[str | None] = mapped_column(String, nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    warnings: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    # Seed used by ``torch.manual_seed`` for the row's current state.
    # See ``ClipLabel.generation_seed`` for rationale.
    generation_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sampled_clip_ids: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
