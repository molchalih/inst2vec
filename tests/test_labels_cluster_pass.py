import json
from dataclasses import dataclass

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from core.config import LabelsSettings
from core.database import (
    Base,
    Clip,
    ClipLabel,
    ClusterLabel,
    User,
    UserCluster,
)
from modules.labels.cluster_pass import run_all_cases


def _labels(**overrides) -> LabelsSettings:
    base = dict(
        case_prompts={"video": "x", "spoken": "x-spoken"},
        cluster_case_prompts={
            "video": "cluster prompt",
            "spoken": "spoken cluster prompt",
        },
    )
    base.update(overrides)
    return LabelsSettings(**base)


@dataclass
class _FakeGen:
    """Stand-in for LabelsGenerator that records calls and emits canned JSON."""

    payloads_by_call: list[str]
    calls: list[tuple[str, int]] = None  # (prompt suffix, max_new_tokens)
    last_schema: dict | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def run_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        schema: dict | None = None,
    ) -> str:
        self.last_schema = schema
        self.calls.append((prompt[-32:], max_new_tokens))
        return self.payloads_by_call.pop(0)

    def run_text_batch(
        self,
        prompts,
        *,
        max_new_tokens,
        seeds,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        schema=None,
    ) -> list[str]:
        return [
            self.run_text(
                p,
                max_new_tokens=max_new_tokens,
                seed=s,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                schema=schema,
            )
            for p, s in zip(prompts, seeds, strict=True)
        ]

    def reclaim_memory(self) -> None:  # pragma: no cover - test stub
        pass


@dataclass
class _RecordingGen:
    """Stand-in that records the FULL prompt body for assertion in tests."""

    payloads_by_call: list[str]
    prompts: list[str] = None

    def __post_init__(self) -> None:
        if self.prompts is None:
            self.prompts = []

    def run_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        schema: dict | None = None,
    ) -> str:
        self.prompts.append(prompt)
        return self.payloads_by_call.pop(0)

    def run_text_batch(
        self,
        prompts,
        *,
        max_new_tokens,
        seeds,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        schema=None,
    ) -> list[str]:
        return [
            self.run_text(
                p,
                max_new_tokens=max_new_tokens,
                seed=s,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                schema=schema,
            )
            for p, s in zip(prompts, seeds, strict=True)
        ]

    def reclaim_memory(self) -> None:  # pragma: no cover - test stub
        pass


def _clean_cluster_json() -> dict:
    return {
        "cluster_label": "soft domestic vignette",
        "cluster_summary": "tight handheld kitchen scenes with warm tungsten tones and shallow focus; domestic intimacy expressed through close framing and a muted personal register",
        "dominant_visual_repertoire": [
            {
                "tag": "warm kitchen",
                "description": "tungsten domestic rooms recurring across the supplied clips",
                "recurrence": "dominant",
            },
            {
                "tag": "shallow focus",
                "description": "blurred backgrounds across recurring shots",
                "recurrence": "frequent",
            },
            {
                "tag": "handheld",
                "description": "subtle frame drift across many clips",
                "recurrence": "frequent",
            },
        ],
        "dominant_aesthetic_logic": [
            {
                "tag": "intimate realism",
                "grounded_in": ["warm kitchen", "handheld"],
                "description": "warm handheld framing reads intimate not staged",
            },
            {
                "tag": "softly tactile",
                "grounded_in": ["warm kitchen", "shallow focus"],
                "description": "warmth and blur combine into a tactile surface",
            },
            {
                "tag": "muted register",
                "grounded_in": ["handheld"],
                "description": "handheld imperfection signals a low-affect personal voice",
            },
        ],
        "taste_signalling": {
            "label": "homecore",
            "description": "the repertoire aligns with a slow-living affinity expressed through domestic intimacy",
            "confidence": "medium",
        },
        "visibility_orientation": {
            "label": "ordinariness",
            "description": "stages attention toward ordinariness over spectacle and polish",
            "confidence": "low",
        },
        "internal_variations": [
            {
                "variation": "bathroom-lit",
                "description": "minor strand of cool-lit grooming clips within the larger repertoire",
            }
        ],
        "boundary_notes": "differs from food-styling clusters by lacking top-down plating and overhead lighting",
        "tool_tags": ["homecore", "warm-palette", "handheld"],
    }


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _seed(eng) -> None:
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(User(id=2, is_selected=True))
        for cid, uid in [(10, 1), (11, 1), (20, 2)]:
            s.add(Clip(id=cid, user_id=uid, is_selected=True, is_downloaded=True))
            s.add(
                ClipLabel(
                    clip_id=cid,
                    label_case="video",
                    status="success",
                    validation="ok",
                    warnings=[],
                    payload={
                        "observable_visual_tags": [{"tag": "x", "evidence": "y"}],
                        "aesthetic_tags": [],
                        "community_signalling_tags": [],
                        "one_sentence_visual_reading": "ok",
                    },
                    attempts=1,
                )
            )
        s.add(
            UserCluster(
                user_id=1,
                embedding_case="video",
                cluster_id=0,
                umap_x=0.0,
                umap_y=0.0,
                centrality=0.9,
            )
        )
        s.add(
            UserCluster(
                user_id=2,
                embedding_case="video",
                cluster_id=0,
                umap_x=0.1,
                umap_y=0.1,
                centrality=0.7,
            )
        )
        s.commit()


def test_cluster_pass_writes_success_row(monkeypatch) -> None:
    eng = _engine()
    _seed(eng)
    fake = _FakeGen(payloads_by_call=[json.dumps(_clean_cluster_json())])
    with Session(eng) as s:
        run_all_cases(
            session=s,
            labels=_labels(),
            generator=fake,
            cases=("video",),
        )
    assert fake.last_schema is not None
    with Session(eng) as s:
        row = s.get(ClusterLabel, ("video", 0))
        assert row is not None
        assert row.status == "success" and row.validation == "ok"
        # The naming pass Title-Case-normalises the provisional label.
        assert row.payload["cluster_label"] == "Soft Domestic Vignette"
        assert isinstance(row.sampled_clip_ids, list) and len(row.sampled_clip_ids) > 0


def test_cluster_hard_validation_fail_retries_with_varying_seed_then_goes_terminal(
    monkeypatch,
) -> None:
    """Cluster-side validation hard-fails (HC1/HC2/HC3) now go through
    ``bump_failure`` because per-attempt seed variation means a retry CAN
    change the output. Each attempt N uses seed ``base + N - 1``. The
    row stays ``pending`` until ``attempts == cluster_max_attempts``,
    then transitions to terminal ``failed``.
    """
    eng = _engine()
    _seed(eng)
    fake = _FakeGen(payloads_by_call=["not json"] * 3)
    labels = _labels(cluster_max_attempts=3, generation_seed=42)
    for _ in range(4):
        with Session(eng) as s:
            run_all_cases(
                session=s,
                labels=labels,
                generator=fake,
                cases=("video",),
            )
    # 3 model calls (one per attempt up to max); the 4th pipeline run
    # finds no pending row (terminal failed) and short-circuits.
    assert len(fake.calls) == 3
    with Session(eng) as s:
        row = s.get(ClusterLabel, ("video", 0))
        assert row is not None
        assert row.status == "failed"
        assert row.attempts == labels.cluster_max_attempts
        # Final attempt's seed lands on the row: base + (max - 1) = 42 + 2.
        assert row.generation_seed == 44


def test_cluster_pass_skips_clusters_with_no_input() -> None:
    eng = _engine()
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(
            UserCluster(
                user_id=1,
                embedding_case="video",
                cluster_id=0,
                umap_x=0.0,
                umap_y=0.0,
                centrality=0.5,
            )
        )
        s.commit()
    fake = _FakeGen(payloads_by_call=[])
    with Session(eng) as s:
        run_all_cases(
            session=s,
            labels=_labels(),
            generator=fake,
            cases=("video",),
        )
    assert fake.calls == []
    with Session(eng) as s:
        row = s.get(ClusterLabel, ("video", 0))
        assert row is not None
        assert row.status == "failed" and row.error == "no_input"
        assert row.attempts == _labels().cluster_max_attempts


def _clean_audio_cluster_json() -> dict:
    payload = dict(_clean_cluster_json())
    payload["dominant_audio_repertoire"] = payload.pop("dominant_visual_repertoire")
    return payload


def test_cluster_pass_spoken_consumes_stage1_clip_labels() -> None:
    """Stage-2 for ``case='spoken'`` now consumes the spoken per-clip
    ``ClipLabel`` payloads written by the stage-1 text pass (mirroring the
    video case). Each member clip's spoken observation tag must reach the
    cluster prompt, while the unrelated video ``ClipLabel`` for the same
    clip must NOT — the spoken cluster pass joins on ``label_case='spoken'``.
    """
    eng = _engine()
    spoken_marker = "spoken-observation-marker"
    spoken_payload = {
        "observable_audio_tags": [{"tag": spoken_marker, "evidence": "0:01"}],
        "aesthetic_tags": [],
        "community_signalling_tags": [],
        "one_sentence_audio_reading": "ok",
    }
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(User(id=2, is_selected=True))
        for cid, uid in [(10, 1), (11, 1), (20, 2)]:
            s.add(Clip(id=cid, user_id=uid, is_selected=True, is_downloaded=True))
            s.add(
                ClipLabel(
                    clip_id=cid,
                    label_case="spoken",
                    status="success",
                    validation="ok",
                    warnings=[],
                    payload=spoken_payload,
                    attempts=1,
                )
            )
            # A visual ClipLabel row that MUST NOT leak into the spoken
            # cluster prompt — the spoken cluster pass joins on label_case.
            s.add(
                ClipLabel(
                    clip_id=cid,
                    label_case="video",
                    status="success",
                    validation="ok",
                    warnings=[],
                    payload={
                        "observable_visual_tags": [
                            {"tag": "visual-only-tag", "evidence": "frames"}
                        ],
                        "aesthetic_tags": [],
                        "community_signalling_tags": [],
                        "one_sentence_visual_reading": "ok",
                    },
                    attempts=1,
                )
            )
        for uid, centrality in [(1, 0.9), (2, 0.7)]:
            s.add(
                UserCluster(
                    user_id=uid,
                    embedding_case="spoken",
                    cluster_id=0,
                    umap_x=0.0,
                    umap_y=0.0,
                    centrality=centrality,
                )
            )
        s.commit()

    fake = _RecordingGen(payloads_by_call=[json.dumps(_clean_audio_cluster_json())])
    with Session(eng) as s:
        run_all_cases(
            session=s,
            labels=_labels(),
            generator=fake,
            cases=("spoken",),
        )

    assert len(fake.prompts) == 1, "expected exactly one cluster prompt for spoken case"
    prompt = fake.prompts[0]
    assert spoken_marker in prompt, (
        "spoken cluster prompt must include each member clip's spoken ClipLabel"
    )
    assert "visual-only-tag" not in prompt, (
        "spoken cluster prompt must NOT include video-case clip labels"
    )

    with Session(eng) as s:
        row = s.get(ClusterLabel, ("spoken", 0))
        assert row is not None and row.status == "success"
        assert "dominant_audio_repertoire" in row.payload


def test_cluster_pass_skips_noise_cluster_id_minus_one() -> None:
    eng = _engine()
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
        s.add(
            ClipLabel(
                clip_id=10,
                status="success",
                validation="ok",
                warnings=[],
                payload={
                    "observable_visual_tags": [{"tag": "x", "evidence": "y"}],
                    "aesthetic_tags": [],
                    "community_signalling_tags": [],
                    "one_sentence_visual_reading": "ok",
                },
                attempts=1,
            )
        )
        s.add(
            UserCluster(
                user_id=1,
                embedding_case="video",
                cluster_id=-1,
                umap_x=0.0,
                umap_y=0.0,
                centrality=0.0,
            )
        )
        s.commit()
    fake = _FakeGen(payloads_by_call=[])
    with Session(eng) as s:
        run_all_cases(
            session=s,
            labels=_labels(),
            generator=fake,
            cases=("video",),
        )
    assert fake.calls == []
    with Session(eng) as s:
        rows = s.execute(select(ClusterLabel)).scalars().all()
        assert rows == []


def test_cluster_pass_wipes_orphan_rows_when_stage_state_missing() -> None:
    """If ``cluster_labels`` carries rows but no ``stage_state`` row exists
    (crash before ``mark_complete``, or restore from backup), the rows
    were produced under an unknown prior fingerprint and must not be
    sealed under the current one. The next pipeline run must wipe them.
    """
    eng = _engine()
    _seed(eng)
    with Session(eng) as s:
        s.add(
            ClusterLabel(
                embedding_case="video",
                cluster_id=0,
                status="success",
                validation="ok",
                warnings=[],
                payload={"cluster_label": "stale", "marker": "old-prompt"},
                attempts=1,
                sampled_clip_ids=[10, 11],
            )
        )
        s.commit()
    fake = _FakeGen(payloads_by_call=[json.dumps(_clean_cluster_json())])
    with Session(eng) as s:
        run_all_cases(
            session=s,
            labels=_labels(),
            generator=fake,
            cases=("video",),
        )
    # The stale "marker" payload must be replaced — proof the orphan row
    # was wiped before the gate re-sealed the fingerprint.
    with Session(eng) as s:
        row = s.get(ClusterLabel, ("video", 0))
        assert row is not None
        # The naming pass Title-Case-normalises the provisional label.
        assert row.payload["cluster_label"] == "Soft Domestic Vignette"
        assert "marker" not in row.payload
    # Generator was actually invoked — proves the row was treated as
    # pending, not as a pre-existing success that the gate would seal.
    assert len(fake.calls) == 1


def _seed_n_clusters(eng, n: int) -> None:
    with Session(eng) as s:
        for cid in range(n):
            uid, clip_id = 100 + cid, 200 + cid
            s.add(User(id=uid, is_selected=True))
            s.add(Clip(id=clip_id, user_id=uid, is_selected=True, is_downloaded=True))
            s.add(
                ClipLabel(
                    clip_id=clip_id,
                    label_case="video",
                    status="success",
                    validation="ok",
                    warnings=[],
                    attempts=1,
                    payload={
                        "observable_visual_tags": [{"tag": "x", "evidence": "y"}],
                        "aesthetic_tags": [],
                        "community_signalling_tags": [],
                        "one_sentence_visual_reading": "ok",
                    },
                )
            )
            s.add(
                UserCluster(
                    user_id=uid,
                    embedding_case="video",
                    cluster_id=cid,
                    umap_x=0.0,
                    umap_y=0.0,
                    centrality=0.9,
                )
            )
        s.commit()


def test_cluster_pass_enforces_unique_labels_within_case() -> None:
    # Per-cluster generation emits the same cluster_label for every cluster and
    # the naming-pass model call returns cluster JSON (useless for naming), so
    # the global naming pass cannot disambiguate via the model. The
    # deterministic exact-uniqueness backstop must still yield distinct labels.
    eng = _engine()
    _seed_n_clusters(eng, 3)
    shared = dict(_clean_cluster_json())
    shared["cluster_label"] = "Shared Name"
    fake = _FakeGen(payloads_by_call=[json.dumps(shared) for _ in range(15)])
    with Session(eng) as s:
        run_all_cases(session=s, labels=_labels(), generator=fake, cases=("video",))
    with Session(eng) as s:
        rows = [s.get(ClusterLabel, ("video", cid)) for cid in range(3)]
        assert all(r is not None and r.status == "success" for r in rows)
        norm = [r.payload["cluster_label"].strip().lower() for r in rows]
        assert len(set(norm)) == 3, f"labels not unique: {norm}"


def test_cluster_pass_dedup_preserves_suffix_for_near_cap_labels() -> None:
    # A shared label already at the 40-char cap: the deterministic backstop must
    # reserve room for its disambiguating token so every label stays distinct
    # AND within the cap (never overflowing or re-colliding).
    eng = _engine()
    _seed_n_clusters(eng, 3)
    shared = dict(_clean_cluster_json())
    shared["cluster_label"] = "A" * 40
    fake = _FakeGen(payloads_by_call=[json.dumps(shared) for _ in range(15)])
    with Session(eng) as s:
        run_all_cases(session=s, labels=_labels(), generator=fake, cases=("video",))
    with Session(eng) as s:
        rows = [s.get(ClusterLabel, ("video", cid)) for cid in range(3)]
        labels = [r.payload["cluster_label"] for r in rows]
        assert len(set(lab.strip().lower() for lab in labels)) == 3, labels
        assert all(len(lab) <= 40 for lab in labels), labels
