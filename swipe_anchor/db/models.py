"""App-store ORM models (plan §3).

All creator keys reference the pipeline's anonymous ``users.id`` as ``creator_id``.
No PII lives here: usernames never cross the export boundary (plan §3, §10).

Portability note: ``JSON`` is used for both jsonb columns and small integer
arrays so the same models run on SQLite (MVP) and Postgres (prod). ``LargeBinary``
backs the bytea ``std_vec``.
"""

from __future__ import annotations

from datetime import datetime

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
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from swipe_anchor.db.base import Base, UTCDateTime

_CREATORS_ID = "creators.creator_id"
_COMPARISONS_ID = "comparisons.comparison_id"
_ANNOTATORS_ID = "annotators.annotator_id"
_ASSIGNMENTS_ID = "assignments.assignment_id"


class AccessCode(Base):
    """A deeplink access code = an annotator identity (plan §7.5 auth).

    The ``code`` is what a person receives in their `?code=…` link; it becomes
    their ``annotator_id`` so every choice + timing links to them. ``note`` is an
    INTERNAL operator annotation (who they are / how you know them) — never shown
    to the annotator and never exported. When this table is empty the backend runs
    open (any non-empty code works, for bootstrap); once any row exists, only
    listed + active codes are admitted.
    """

    __tablename__ = "access_codes"

    code: Mapped[str] = mapped_column(String, primary_key=True)
    note: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Creator(Base):
    __tablename__ = "creators"

    creator_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    seed_cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    seed_group: Mapped[str | None] = mapped_column(String, nullable=True)
    digest_version: Mapped[int] = mapped_column(Integer, default=1)
    std_vec: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    exported_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class CreatorDigest(Base):
    __tablename__ = "creator_digests"

    creator_id: Mapped[int] = mapped_column(ForeignKey(_CREATORS_ID), primary_key=True)
    digest_version: Mapped[int] = mapped_column(Integer, primary_key=True)
    rep_clip_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    caption_keywords: Mapped[dict] = mapped_column(JSON, default=dict)
    audio_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    n_clips: Mapped[int] = mapped_column(Integer, default=0)

    creator = relationship("Creator")


class DigestClip(Base):
    __tablename__ = "digest_clips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    creator_id: Mapped[int] = mapped_column(ForeignKey(_CREATORS_ID))
    clip_id: Mapped[int] = mapped_column(Integer)
    ord: Mapped[int] = mapped_column(Integer, default=0)
    video_url: Mapped[str | None] = mapped_column(String, nullable=True)
    poster_url: Mapped[str | None] = mapped_column(String, nullable=True)
    audio_url: Mapped[str | None] = mapped_column(String, nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_medoid: Mapped[bool] = mapped_column(Boolean, default=False)

    creator = relationship("Creator")

    __table_args__ = (UniqueConstraint("creator_id", "clip_id"),)


class Annotator(Base):
    __tablename__ = "annotators"

    annotator_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    reliability: Mapped[float] = mapped_column(Float, default=0.5)
    n_responses: Mapped[int] = mapped_column(Integer, default=0)
    n_gold_seen: Mapped[int] = mapped_column(Integer, default=0)
    n_gold_correct: Mapped[int] = mapped_column(Integer, default=0)
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False)
    last_seen: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Comparison(Base):
    __tablename__ = "comparisons"

    comparison_id: Mapped[str] = mapped_column(String, primary_key=True)
    creator_a: Mapped[int] = mapped_column(Integer)
    creator_b: Mapped[int] = mapped_column(Integer)
    creator_c: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String, default="random")
    seed_group: Mapped[str | None] = mapped_column(String, nullable=True)
    expected_modality: Mapped[str | None] = mapped_column(String, nullable=True)
    target_k: Mapped[int] = mapped_column(Integer, default=5)
    n_judgments: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String, default="open")
    information: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    @property
    def creators(self) -> tuple[int, int, int]:
        return (self.creator_a, self.creator_b, self.creator_c)


class GoldItem(Base):
    __tablename__ = "gold_items"

    comparison_id: Mapped[str] = mapped_column(
        ForeignKey(_COMPARISONS_ID), primary_key=True
    )
    known_odd: Mapped[int] = mapped_column(Integer)
    difficulty: Mapped[str] = mapped_column(String, default="obvious")

    comparison = relationship("Comparison")


class Assignment(Base):
    __tablename__ = "assignments"

    assignment_id: Mapped[str] = mapped_column(String, primary_key=True)
    comparison_id: Mapped[str] = mapped_column(ForeignKey(_COMPARISONS_ID))
    annotator_id: Mapped[str] = mapped_column(ForeignKey(_ANNOTATORS_ID))
    issued_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )
    status: Mapped[str] = mapped_column(String, default="issued")

    comparison = relationship("Comparison")
    annotator = relationship("Annotator")

    __table_args__ = (
        UniqueConstraint("comparison_id", "annotator_id", name="uq_assignment_once"),
    )


class Response(Base):
    __tablename__ = "responses"

    response_id: Mapped[str] = mapped_column(String, primary_key=True)
    assignment_id: Mapped[str] = mapped_column(ForeignKey(_ASSIGNMENTS_ID))
    comparison_id: Mapped[str] = mapped_column(ForeignKey(_COMPARISONS_ID))
    annotator_id: Mapped[str] = mapped_column(ForeignKey(_ANNOTATORS_ID))
    odd_creator_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reaction_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    card_dwell_ms: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    expanded: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    client_meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Which reel was actually shown per creator: {creator_id: clip_id}. The card
    # renders the lowest-ord digest clip, so historical rows can be backfilled.
    shown_clips: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    assignment = relationship("Assignment")
    comparison = relationship("Comparison")
    annotator = relationship("Annotator")

    # At most one judgment per assignment — the DB-level backstop to the atomic
    # claim in record_response (concurrent double-submit cannot double-count).
    __table_args__ = (
        UniqueConstraint("assignment_id", name="uq_response_per_assignment"),
    )


class Triplet(Base):
    """Derived ordinal constraint: two per RESOLVED comparison, materialized from
    the consensus odd-one-out (not per-response). ``response_id`` is retained as a
    nullable column for provenance but is left NULL by consensus materialization."""

    __tablename__ = "triplets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    anchor_id: Mapped[int] = mapped_column(Integer)
    positive_id: Mapped[int] = mapped_column(Integer)
    negative_id: Mapped[int] = mapped_column(Integer)
    comparison_id: Mapped[str] = mapped_column(ForeignKey(_COMPARISONS_ID))
    response_id: Mapped[str | None] = mapped_column(
        ForeignKey("responses.response_id"), nullable=True
    )
    weight: Mapped[float] = mapped_column(Float, default=1.0)

    comparison = relationship("Comparison")
    response = relationship("Response")


class Consensus(Base):
    __tablename__ = "consensus"

    comparison_id: Mapped[str] = mapped_column(
        ForeignKey(_COMPARISONS_ID), primary_key=True
    )
    consensus_odd: Mapped[int | None] = mapped_column(Integer, nullable=True)
    agreement: Mapped[float] = mapped_column(Float, default=0.0)
    n_effective: Mapped[float] = mapped_column(Float, default=0.0)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)

    comparison = relationship("Comparison")


class ReliabilityEvent(Base):
    """Append-only audit trail for the Dawid-Skene model (plan §3)."""

    __tablename__ = "reliability_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    annotator_id: Mapped[str] = mapped_column(ForeignKey(_ANNOTATORS_ID))
    comparison_id: Mapped[str] = mapped_column(ForeignKey(_COMPARISONS_ID))
    agreed_with_consensus: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    gold_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    ts: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    annotator = relationship("Annotator")
    comparison = relationship("Comparison")


class Reminder(Base):
    """A scheduled "come back" DM for a Telegram user (the rest-nudge follow-up).

    Created when someone taps "remind me later" in the Mini App; the bot polls for
    due rows and sends the message. ``telegram_id`` is the raw chat id (recovered
    from validated initData) — this is the one place it is stored, and only to be
    able to message the person back.
    """

    __tablename__ = "reminders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Raw Telegram chat id: 64-bit (BIGINT on Postgres) — modern ids exceed 2^31.
    telegram_id: Mapped[int] = mapped_column(BigInteger, index=True)
    due_at: Mapped[datetime] = mapped_column(UTCDateTime())
    sent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


__all__ = [
    "AccessCode",
    "Annotator",
    "Assignment",
    "Comparison",
    "Consensus",
    "Creator",
    "CreatorDigest",
    "DigestClip",
    "GoldItem",
    "ReliabilityEvent",
    "Reminder",
    "Response",
    "Triplet",
]
