"""FastAPI app for the swipe-anchor backend (plan §2.1, §8.3).

Mirrors ``services/atlas_api`` conventions: a ``build_app`` factory with injected
``session_factory`` / ``token`` / ``cors_origin`` closures, bearer-token auth, and
a narrow JSON contract. Endpoints stay thin — all logic lives in
``swipe_anchor.backend.service`` so it is unit-testable without HTTP.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from swipe_anchor.backend.service import (
    ForbiddenError,
    InvalidOddCreatorError,
    UnknownAssignmentError,
    is_access_allowed,
    next_batch,
    record_response,
)
from swipe_anchor.backend.tg_auth import code_for_telegram_id, validate_init_data
from swipe_anchor.config import Settings
from swipe_anchor.db.models import Assignment, CreatorDigest, DigestClip

SessionFactory = Callable[[], AbstractContextManager[Session]]

# Per-event activity log: who (access code) chose what, and how long it took.
log = logging.getLogger("swipe_anchor.activity")


class NextBatchRequest(BaseModel):
    n: int = 3


class ClipMedia(BaseModel):
    clip_id: int
    video_url: str | None = None
    poster_url: str | None = None


class CreatorCard(BaseModel):
    creator_id: int
    seed_group: str | None = None
    rep_clip_ids: list[int] = []
    caption_keywords: dict = {}
    audio_summary: dict = {}
    clips: list[ClipMedia] = []


class BatchItem(BaseModel):
    assignment_id: str
    comparison_id: str
    seed_group: str | None = None
    expected_modality: str | None = None
    creators: list[CreatorCard]


class NextBatchResponse(BaseModel):
    items: list[BatchItem]


class RespondRequest(BaseModel):
    assignment_id: str
    odd_creator_id: int | None = None
    confidence: float = 1.0
    reaction_time_ms: int | None = None
    card_dwell_ms: dict | None = None
    shown_clips: dict | None = None
    expanded: bool = False


class RespondResponse(BaseModel):
    accepted: bool
    n_triplets: int
    retired: bool


class TgAuthRequest(BaseModel):
    init_data: str


class TgRegisterRequest(BaseModel):
    telegram_id: int
    name: str | None = None


class TgExistsRequest(BaseModel):
    telegram_id: int


class TgRemindRequest(BaseModel):
    init_data: str


# How far out a "remind me later" tap schedules the come-back DM.
REMIND_HOURS = 20.0


def _digest_card(
    session: Session, creator_id: int, seed_group: str | None
) -> CreatorCard:
    """Build a creator card from the stored digest, degrading gracefully.

    Before the export job has run (Phase 0), digests are absent; the card still
    carries the creator id so the loop is exercisable end to end.
    """
    digest = (
        session.query(CreatorDigest)
        .filter_by(creator_id=creator_id)
        .order_by(CreatorDigest.digest_version.desc())
        .first()
    )
    if digest is None:
        return CreatorCard(creator_id=creator_id, seed_group=seed_group)
    clip_rows = (
        session.query(DigestClip)
        .filter_by(creator_id=creator_id)
        .order_by(DigestClip.ord)
        .all()
    )
    return CreatorCard(
        creator_id=creator_id,
        seed_group=seed_group,
        rep_clip_ids=list(digest.rep_clip_ids or []),
        caption_keywords=digest.caption_keywords or {},
        audio_summary=digest.audio_summary or {},
        clips=[
            ClipMedia(clip_id=c.clip_id, video_url=c.video_url, poster_url=c.poster_url)
            for c in clip_rows
        ],
    )


def build_app(
    session_factory: SessionFactory,
    *,
    token: str = "",
    cors_origin: str = "",
    frontend_dist: str | None = None,
    media_dir: str | None = None,
    settings: Settings | None = None,
    tg_bot_token: str = "",
    tg_internal_token: str = "",
) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="swipe-anchor", version="0.1.0")

    if cors_origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[cors_origin],
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    def verify_token(authorization: str | None = Header(default=None)) -> None:
        if not token:
            return
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    def access_code(session: Session, x_access_code: str | None) -> str:
        """Resolve + admit the per-user access code (the annotator identity).

        Missing code -> 401 (gate the app); present-but-not-admitted -> 403.
        """
        code = (x_access_code or "").strip()
        if not code:
            raise HTTPException(status_code=401, detail="access code required")
        if not is_access_allowed(session, code):
            raise HTTPException(status_code=403, detail="access code not recognised")
        return code

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/next-batch", response_model=NextBatchResponse)
    def post_next_batch(
        req: NextBatchRequest,
        _: None = Depends(verify_token),
        x_access_code: str | None = Header(default=None, alias="X-Access-Code"),
    ) -> NextBatchResponse:
        with session_factory() as session:
            code = access_code(session, x_access_code)
            picks = next_batch(session, code, req.n, settings=settings)
            session.flush()
            items: list[BatchItem] = []
            for cmp in picks:
                asg = (
                    session.query(Assignment)
                    .filter_by(annotator_id=code, comparison_id=cmp.comparison_id)
                    .one()
                )
                items.append(
                    BatchItem(
                        assignment_id=asg.assignment_id,
                        comparison_id=cmp.comparison_id,
                        seed_group=cmp.seed_group,
                        expected_modality=cmp.expected_modality,
                        creators=[
                            _digest_card(session, cid, cmp.seed_group)
                            for cid in cmp.creators
                        ],
                    )
                )
            log.info("issue code=%s n=%d", code, len(items))
            return NextBatchResponse(items=items)

    @app.post("/respond", response_model=RespondResponse)
    def post_respond(
        req: RespondRequest,
        _: None = Depends(verify_token),
        x_access_code: str | None = Header(default=None, alias="X-Access-Code"),
    ) -> RespondResponse:
        with session_factory() as session:
            code = access_code(session, x_access_code)
            try:
                result = record_response(
                    session,
                    req.assignment_id,
                    odd_id=req.odd_creator_id,
                    confidence=req.confidence,
                    reaction_time_ms=req.reaction_time_ms,
                    card_dwell_ms=req.card_dwell_ms,
                    shown_clips=req.shown_clips,
                    expanded=req.expanded,
                    expected_annotator_id=code,
                    settings=settings,
                )
            except UnknownAssignmentError:
                raise HTTPException(
                    status_code=404, detail="unknown assignment"
                ) from None
            except ForbiddenError:
                raise HTTPException(
                    status_code=403, detail="assignment not owned by this code"
                ) from None
            except InvalidOddCreatorError:
                raise HTTPException(
                    status_code=422, detail="odd_creator_id not in comparison"
                ) from None
            log.info(
                "answer code=%s assignment=%s odd=%s rt_ms=%s dwell_ms=%s "
                "triplets=%d accepted=%s",
                code,
                req.assignment_id,
                req.odd_creator_id,
                req.reaction_time_ms,
                req.card_dwell_ms,
                result.n_triplets,
                result.accepted,
            )
            return RespondResponse(
                accepted=result.accepted,
                n_triplets=result.n_triplets,
                retired=result.retired,
            )

    @app.get("/metrics")
    def metrics(_: None = Depends(verify_token)) -> dict:
        from sqlalchemy import func

        from swipe_anchor.backend.consensus_writer import classes_for_comparison
        from swipe_anchor.core.consensus import Vote, agreement
        from swipe_anchor.db.models import (
            Annotator,
            Comparison,
            Response,
            Triplet,
        )

        with session_factory() as session:

            def count(model, **f) -> int:
                q = session.query(func.count()).select_from(model)
                return int(q.filter_by(**f).scalar() if f else q.scalar())

            # Agreement over resolved (retired) + ambiguous items; skip is its own category.
            items: dict[str, list[Vote]] = {}
            resolved = session.query(Comparison).filter(
                Comparison.status.in_(("retired", "ambiguous"))
            )
            for cmp in resolved:
                order = classes_for_comparison(cmp)
                class_of = {cid: i for i, cid in enumerate(order)}
                votes = [
                    Vote(
                        r.annotator_id,
                        None if r.odd_creator_id is None else class_of[r.odd_creator_id],
                    )
                    for r in session.query(Response).filter_by(comparison_id=cmp.comparison_id)
                ]
                if votes:
                    items[cmp.comparison_id] = votes
            rels = [a.reliability for a in session.query(Annotator)]
            return {
                "n_annotators": count(Annotator),
                "mean_reliability": (sum(rels) / len(rels)) if rels else 0.0,
                "comparisons": {
                    "open": count(Comparison, status="open"),
                    "retired": count(Comparison, status="retired"),
                    "ambiguous": count(Comparison, status="ambiguous"),
                    "gold": count(Comparison, status="gold"),
                },
                "triplets": count(Triplet),
                "agreement": agreement(items),
            }

    if tg_bot_token:

        def verify_internal(
            x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
        ) -> None:
            if not tg_internal_token or x_internal_token != tg_internal_token:
                raise HTTPException(status_code=401, detail="bad internal token")

        @app.post("/tg/auth")
        def tg_auth(req: TgAuthRequest) -> dict[str, str]:
            user = validate_init_data(req.init_data, tg_bot_token)
            if user is None:
                raise HTTPException(status_code=401, detail="invalid init data")
            code = code_for_telegram_id(user.id, tg_bot_token)
            with session_factory() as session:
                from swipe_anchor.db.models import AccessCode

                row = session.get(AccessCode, code)
                if row is None or not row.is_active:
                    raise HTTPException(status_code=403, detail="not registered")
            log.info("tg_auth ok tg_id=%s code=%s", user.id, code)
            return {"access_code": code}

        @app.post("/tg/register")
        def tg_register(
            req: TgRegisterRequest,
            _: None = Depends(verify_internal),
        ) -> dict[str, object]:
            from swipe_anchor.db.models import AccessCode

            code = code_for_telegram_id(req.telegram_id, tg_bot_token)
            with session_factory() as session:
                row = session.get(AccessCode, code)
                if row is None:
                    session.add(
                        AccessCode(code=code, note=req.name, is_active=True)
                    )
                else:
                    if req.name is not None:
                        row.note = req.name
                    row.is_active = True
            log.info("tg_register tg_id=%s code=%s", req.telegram_id, code)
            return {"ok": True, "access_code": code}

        @app.post("/tg/exists")
        def tg_exists(
            req: TgExistsRequest,
            _: None = Depends(verify_internal),
        ) -> dict[str, object]:
            from swipe_anchor.db.models import AccessCode

            code = code_for_telegram_id(req.telegram_id, tg_bot_token)
            with session_factory() as session:
                row = session.get(AccessCode, code)
                exists = row is not None and bool(row.is_active)
            return {"exists": exists, "access_code": code}

        @app.post("/tg/stats")
        def tg_stats(_: None = Depends(verify_internal)) -> dict:
            from swipe_anchor.backend.stats import gather_stats

            with session_factory() as session:
                return gather_stats(session)

        @app.post("/tg/remind")
        def tg_remind(req: TgRemindRequest) -> dict[str, object]:
            # Authenticated by the signed initData (same as /tg/auth) — that's how
            # we recover the raw telegram id to message later.
            user = validate_init_data(req.init_data, tg_bot_token)
            if user is None:
                raise HTTPException(status_code=401, detail="invalid init data")
            from swipe_anchor.backend.reminders import schedule_reminder

            with session_factory() as session:
                due = schedule_reminder(session, user.id, hours=REMIND_HOURS)
            log.info("tg_remind tg_id=%s due=%s", user.id, due.isoformat())
            return {"ok": True, "due_at": due.isoformat()}

        @app.post("/tg/due-reminders")
        def tg_due_reminders(_: None = Depends(verify_internal)) -> dict[str, object]:
            from swipe_anchor.backend.reminders import pop_due_reminders

            with session_factory() as session:
                ids = pop_due_reminders(session)
            return {"telegram_ids": ids}

    # Static mounts come after the API routes so those always win. `/media`
    # (creator reels + posters from data/source) is more specific than the SPA
    # catch-all at `/`, so it must be registered first.
    if media_dir:
        from fastapi.staticfiles import StaticFiles

        app.mount("/media", StaticFiles(directory=media_dir), name="media")

    if frontend_dist:
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="ui")

    return app
