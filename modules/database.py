import os

from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship
from sqlalchemy.sql import func

load_dotenv()

# create an instance of the ORM engine
engine = create_engine(os.environ["DATABASE_URL"])

# create a base class for the ORM models
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    pk = Column(BigInteger, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String)
    profile_pic_url = Column(String)
    profile_pic_url_hd = Column(String)
    following_count = Column(Integer)
    city_name = Column(String)
    user_disqualified = Column(Integer, nullable=True)
    parse_status = Column(String, nullable=True)

    clips = relationship("Clip", back_populates="user")
    embeddings = relationship("UserEmbedding", back_populates="user")
    clusters = relationship("UserCluster", back_populates="user")


class Clip(Base):
    __tablename__ = "clips"

    pk = Column(BigInteger, primary_key=True)
    user_pk = Column(BigInteger, ForeignKey("users.pk"), nullable=False)
    thumbnail_url = Column(String)
    video_url = Column(String)
    caption_text = Column(Text)
    caption_language = Column(Text, nullable=True)
    caption_translation = Column(Text)
    comment_count = Column(Integer)
    reshare_count = Column(Integer)
    like_count = Column(Integer)
    play_count = Column(Integer)
    music_id = Column(Integer, ForeignKey("music.id"), nullable=True)
    music_confidence = Column(Float)
    has_music = Column(Integer, nullable=True)
    speech_transcription = Column(Text)
    speech_language = Column(String, nullable=True)
    speech_confidence = Column(Float, nullable=True)
    speech_avg_logprob = Column(Float, nullable=True)
    speech_compression_ratio = Column(Float, nullable=True)
    has_speech = Column(Integer, nullable=True)
    speech_translation = Column(Text)
    disqualified = Column(Integer, nullable=True)

    user = relationship("User", back_populates="clips")
    music = relationship("Music", back_populates="clips")
    embeddings = relationship("ClipEmbedding", back_populates="clip")


class Music(Base):
    __tablename__ = "music"
    __table_args__ = (
        UniqueConstraint("artist", "track", name="uq_music_artist_track"),
    )  # additional enforcement of artists and tracks

    id = Column(Integer, primary_key=True, autoincrement=True)
    artist = Column(String, nullable=False, default="")
    track = Column(String, nullable=False, default="")
    spotify_id = Column(String, nullable=True)
    reccobeats_id = Column(String, nullable=True)
    has_features = Column(String, nullable=True)
    acousticness = Column(Float, nullable=True)
    danceability = Column(Float, nullable=True)
    energy = Column(Float, nullable=True)
    instrumentalness = Column(Float, nullable=True)
    key = Column(Integer, nullable=True)
    liveness = Column(Float, nullable=True)
    loudness = Column(Float, nullable=True)
    mode = Column(Integer, nullable=True)
    speechiness = Column(Float, nullable=True)
    tempo = Column(Float, nullable=True)
    valence = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    clips = relationship("Clip", back_populates="music")


class Download(Base):
    __tablename__ = "downloads"

    entity_pk = Column(BigInteger, primary_key=True)
    file_type = Column(String, primary_key=True)
    success = Column(Boolean)
    # Legacy; unused by fetch_profiles / parse_state (see users.parse_status).
    parse_available = Column(Boolean, default=True)


class ClipEmbedding(Base):
    __tablename__ = "clip_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "clip_pk", "embedding_case", name="uq_clip_embeddings_clip_case"
        ),
    )

    clip_pk = Column(BigInteger, ForeignKey("clips.pk"), primary_key=True)
    embedding_case = Column(String, primary_key=True)
    embedding = Column(LargeBinary, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    clip = relationship("Clip", back_populates="embeddings")


class UserEmbedding(Base):
    __tablename__ = "user_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "user_pk", "embedding_case", name="uq_user_embeddings_user_case"
        ),
    )

    user_pk = Column(BigInteger, ForeignKey("users.pk"), primary_key=True)
    embedding_case = Column(String, primary_key=True)
    embedding = Column(LargeBinary, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user = relationship("User", back_populates="embeddings")


class UserCluster(Base):
    __tablename__ = "user_clusters"
    __table_args__ = (
        UniqueConstraint(
            "user_pk", "embedding_case", name="uq_user_clusters_user_case"
        ),
    )

    user_pk = Column(BigInteger, ForeignKey("users.pk"), primary_key=True)
    embedding_case = Column(String, primary_key=True)
    cluster_id = Column(Integer, nullable=False)
    umap_x = Column(Float, nullable=False)
    umap_y = Column(Float, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    user = relationship("User", back_populates="clusters")


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

    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding_case = Column(String, nullable=False)
    umap_n_components = Column(Integer, nullable=False)
    umap_n_neighbors = Column(Integer, nullable=False)
    umap_min_dist = Column(Float, nullable=False)
    umap_metric = Column(String, nullable=False)
    umap2d_n_neighbors = Column(Integer, nullable=False)
    umap2d_min_dist = Column(Float, nullable=False)
    umap2d_metric = Column(String, nullable=False)
    hdbscan_min_cluster_size = Column(Integer, nullable=False)
    hdbscan_min_samples = Column(Integer, nullable=True)
    hdbscan_cluster_selection_method = Column(String, nullable=False)
    hdbscan_metric = Column(String, nullable=False)
    random_state = Column(Integer, nullable=False)
    n_clusters = Column(Integer, nullable=False)
    noise_ratio = Column(Float, nullable=False)
    min_size = Column(Integer, nullable=False)
    median_size = Column(Integer, nullable=False)
    max_size = Column(Integer, nullable=False)
    disqualified = Column(Integer, nullable=True)
    dbcv = Column(Float, nullable=True)
    silhouette = Column(Float, nullable=True)
    param_plateau_score = Column(Float, nullable=True)
    in_current_grid = Column(Integer, nullable=True)  # 1=current, 0=stale
    dataset_hash = Column(String, nullable=True)  # SHA-256 of sorted user PKs
    validation_config_hash = Column(String, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


def init_db():
    """create the database."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
