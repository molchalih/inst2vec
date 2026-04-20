import csv
import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    Column,
    BigInteger,
    Integer,
    Float,
    String,
    Boolean,
    ForeignKey,
    Text,
    LargeBinary,
    UniqueConstraint,
    DateTime,
)
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy.sql import func
from sqlalchemy import inspect, text

from modules.services import log

load_dotenv()

engine = create_engine(os.environ["DATABASE_URL"])
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
    )

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
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
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
    parse_available = Column(Boolean, default=True)


class ClipEmbedding(Base):
    __tablename__ = "clip_embeddings"
    __table_args__ = (
        UniqueConstraint("clip_pk", "embedding_case", name="uq_clip_embeddings_clip_case"),
    )

    clip_pk = Column(BigInteger, ForeignKey("clips.pk"), primary_key=True)
    embedding_case = Column(String, primary_key=True)
    embedding = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
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
        UniqueConstraint("user_pk", "embedding_case", name="uq_user_embeddings_user_case"),
    )

    user_pk = Column(BigInteger, ForeignKey("users.pk"), primary_key=True)
    embedding_case = Column(String, primary_key=True)
    embedding = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
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
        UniqueConstraint("user_pk", "embedding_case", name="uq_user_clusters_user_case"),
    )

    user_pk = Column(BigInteger, ForeignKey("users.pk"), primary_key=True)
    embedding_case = Column(String, primary_key=True)
    cluster_id = Column(Integer, nullable=False)
    umap_x = Column(Float, nullable=False)
    umap_y = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
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
            "umap_n_components", "umap_n_neighbors", "umap_min_dist", "umap_metric",
            "umap2d_n_neighbors", "umap2d_min_dist", "umap2d_metric",
            "hdbscan_min_cluster_size", "hdbscan_min_samples",
            "hdbscan_cluster_selection_method", "hdbscan_metric",
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
    in_current_grid = Column(Integer, nullable=True)   # 1=current, 0=stale
    dataset_hash = Column(String, nullable=True) # SHA-256 of sorted user PKs
    validation_config_hash = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def _migrate_clips_table() -> None:
    """Apply additive schema migrations for existing SQLite databases."""
    inspector = inspect(engine)
    if "clips" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("clips")}
    if "caption_language" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE clips ADD COLUMN caption_language TEXT"))
    if "disqualified" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE clips ADD COLUMN disqualified INTEGER"))


def _migrate_users_table() -> None:
    """Apply additive schema migrations for existing users table."""
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("users")}
    if "user_disqualified" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN user_disqualified INTEGER"))


def _migrate_cluster_runs_table() -> None:
    """Apply additive schema migrations for existing cluster_runs table."""
    inspector = inspect(engine)
    if "cluster_runs" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("cluster_runs")}
    new_cols = {
        "disqualified": "INTEGER",
        "dbcv": "REAL",
        "silhouette": "REAL",
        "param_plateau_score": "REAL",
        "in_current_grid": "INTEGER",
        "dataset_hash": "TEXT",
        "validation_config_hash": "TEXT",
    }
    for col, col_type in new_cols.items():
        if col not in columns:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE cluster_runs ADD COLUMN {col} {col_type}"))

    # Backfill renamed column for existing DBs that still have dataset_fingerprint.
    if "dataset_hash" in {c["name"] for c in inspect(engine).get_columns("cluster_runs")} and "dataset_fingerprint" in columns:
        with engine.begin() as conn:
            conn.execute(text(
                "UPDATE cluster_runs "
                "SET dataset_hash = dataset_fingerprint "
                "WHERE dataset_hash IS NULL AND dataset_fingerprint IS NOT NULL"
            ))


# legacy column drops run during init_db; no separate migrate script needed for these.
def _migrate_cluster_runs_drop_legacy_columns() -> None:
    """Drop unused legacy columns from cluster_runs (SQLite 3.35+ ALTER DROP COLUMN)."""
    inspector = inspect(engine)
    if "cluster_runs" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("cluster_runs")}
    for col in ("composite_score", "bootstrap_stability", "bootstrap_n_runs"):
        if col not in columns:
            continue
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE cluster_runs DROP COLUMN {col}"))
        except Exception as exc:
            log(
                "database",
                f"could not drop cluster_runs.{col} (needs sqlite 3.35+): {exc}",
                level="warn",
            )


def init_db():
    Base.metadata.create_all(engine)
    _migrate_users_table()
    _migrate_clips_table()
    _migrate_cluster_runs_table()
    _migrate_cluster_runs_drop_legacy_columns()


def get_session() -> Session:
    return Session(engine)


def load_usernames_from_csv(csv_path: str = "data/data.csv"):
    with open(csv_path) as f:
        reader = csv.reader(f)
        urls = [row[0].strip() for row in reader if row]

    usernames = set()
    for url in urls:
        path = urlparse(url).path.strip("/")
        if path:
            # handle trailing /reels/ etc — take first segment
            username = path.split("/")[0]
            if username:
                usernames.add(username)

    total = len(urls)
    unique = len(usernames)
    duplicates_in_csv = total - unique

    session = get_session()
    loaded = 0
    for username in sorted(usernames):
        if not session.query(User).filter_by(username=username).first():
            session.add(User(pk=hash(username) & 0x7FFFFFFFFFFFFFFF, username=username))
            loaded += 1
    session.commit()
    already_in_db = unique - loaded
    log("database", f"loaded {loaded} usernames ({duplicates_in_csv} duplicates in csv, {already_in_db} already in db)")
    session.close()
