from typing import Optional

from sqlalchemy import (
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
    create_engine,
    text,
)
from sqlalchemy.engine import Engine as _Engine
from sqlalchemy.orm import (
    DeclarativeBase,  # type: ignore[name-defined]
    Mapped,
    Session,
    mapped_column,  # type: ignore[name-defined]
    relationship,
)
from sqlalchemy.sql import func

_engine: _Engine | None = None


def get_engine() -> _Engine:
    assert _engine is not None, "Call init_db() before using the database"
    return _engine


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    following_count: Mapped[int | None] = mapped_column(Integer)
    follower_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parse_status: Mapped[str | None] = mapped_column(String, nullable=True)
    is_low_plays_median: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_not_enough_clips: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
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
    caption_language: Mapped[str | None] = mapped_column(Text, nullable=True)
    caption_translation: Mapped[str | None] = mapped_column(Text)
    comment_count: Mapped[int | None] = mapped_column(Integer)
    reshare_count: Mapped[int | None] = mapped_column(Integer)
    like_count: Mapped[int | None] = mapped_column(Integer)
    play_count: Mapped[int | None] = mapped_column(Integer)
    video_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    taken_at: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    music_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("music.id"), nullable=True
    )
    music_confidence: Mapped[float | None] = mapped_column(Float)
    is_music_recognized: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    speech_transcription: Mapped[str | None] = mapped_column(Text)
    speech_language: Mapped[str | None] = mapped_column(String, nullable=True)
    speech_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    speech_avg_logprob: Mapped[float | None] = mapped_column(Float, nullable=True)
    speech_compression_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_speech_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    has_speech: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    speech_translation: Mapped[str | None] = mapped_column(Text)
    is_garbage: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_too_short: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_too_long: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_too_old: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_low_percentile: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_high_percentile: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_creator_low_outlier: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    log_plays: Mapped[float | None] = mapped_column(Float, nullable=True)
    creator_relative_robust_z: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    is_preprocessed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_eligible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_selected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_downloaded: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="clips")  # type: ignore[assignment]
    music: Mapped[Optional["Music"]] = relationship("Music", back_populates="clips")  # type: ignore[assignment]
    embeddings: Mapped[list["ClipEmbedding"]] = relationship(  # type: ignore[assignment]
        "ClipEmbedding", back_populates="clip"
    )


class Music(Base):
    __tablename__ = "music"
    __table_args__ = (
        UniqueConstraint("artist", "track", name="uq_music_artist_track"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artist: Mapped[str] = mapped_column(String, nullable=False, default="")
    track: Mapped[str] = mapped_column(String, nullable=False, default="")
    spotify_id: Mapped[str | None] = mapped_column(String, nullable=True)
    reccobeats_id: Mapped[str | None] = mapped_column(String, nullable=True)
    is_audio_features_extracted: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )
    acousticness: Mapped[float | None] = mapped_column(Float, nullable=True)
    danceability: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy: Mapped[float | None] = mapped_column(Float, nullable=True)
    instrumentalness: Mapped[float | None] = mapped_column(Float, nullable=True)
    key: Mapped[int | None] = mapped_column(Integer, nullable=True)
    liveness: Mapped[float | None] = mapped_column(Float, nullable=True)
    loudness: Mapped[float | None] = mapped_column(Float, nullable=True)
    mode: Mapped[int | None] = mapped_column(Integer, nullable=True)
    speechiness: Mapped[float | None] = mapped_column(Float, nullable=True)
    tempo: Mapped[float | None] = mapped_column(Float, nullable=True)
    valence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    clips: Mapped[list["Clip"]] = relationship("Clip", back_populates="music")  # type: ignore[assignment]


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
    eligibility: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    dbcv: Mapped[float | None] = mapped_column(Float, nullable=True)
    silhouette: Mapped[float | None] = mapped_column(Float, nullable=True)
    param_plateau_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    in_current_grid: Mapped[bool | None] = mapped_column(
        Boolean, nullable=True
    )  # True=current, False=stale
    dataset_hash: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # SHA-256 of sorted user PKs
    validation_config_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


def init_db(database_url: str, identity_db_url: str) -> None:
    global _engine
    from modules.identity import init_identity_db

    _engine = create_engine(database_url)
    Base.metadata.create_all(_engine)
    init_identity_db(identity_db_url)


def get_session() -> Session:
    return Session(get_engine())


def clip_used_in_analysis():
    """Canonical filter: clips that should drive downstream computation.

    Returns a tuple of clauses for `query.filter(*clip_used_in_analysis())`.
    """
    return (
        Clip.is_selected.is_(True),
        Clip.is_downloaded.is_(True),
    )
