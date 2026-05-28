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
        case_prompts={"video": "x", "audio": "x-audio"},
        cluster_case_prompts={
            "video": "cluster prompt",
            "audio": "audio cluster prompt",
        },
    )
    base.update(overrides)
    return LabelsSettings(**base)


@dataclass
class _FakeGen:
    """Stand-in for LabelsGenerator that records calls and emits canned JSON."""

    payloads_by_call: list[str]
    calls: list[tuple[str, int]] = None  # (prompt suffix, max_new_tokens)

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def run_text(self, prompt: str, *, max_new_tokens: int) -> str:
        self.calls.append((prompt[-32:], max_new_tokens))
        return self.payloads_by_call.pop(0)


@dataclass
class _RecordingGen:
    """Stand-in that records the FULL prompt body for assertion in tests."""

    payloads_by_call: list[str]
    prompts: list[str] = None

    def __post_init__(self) -> None:
        if self.prompts is None:
            self.prompts = []

    def run_text(self, prompt: str, *, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        return self.payloads_by_call.pop(0)


def _clean_cluster_json() -> dict:
    return {
        "cluster_label": "soft domestic vignette",
        "cluster_summary": "tight handheld kitchen scenes with warm tones",
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
    with Session(eng) as s:
        row = s.get(ClusterLabel, ("video", 0))
        assert row is not None
        assert row.status == "success" and row.validation == "ok"
        assert row.payload["cluster_label"] == "soft domestic vignette"
        assert isinstance(row.sampled_clip_ids, list) and len(row.sampled_clip_ids) > 0


def test_cluster_pass_retries_then_fails(monkeypatch) -> None:
    """Retry semantics are per-pipeline-run: each ``run_all_cases`` call
    consumes at most one attempt per cluster; the row transitions to
    ``failed`` once ``attempts`` reaches ``cluster_max_attempts``.
    """
    eng = _engine()
    _seed(eng)
    fake = _FakeGen(payloads_by_call=["not json", "not json", "not json"])
    labels = _labels(cluster_max_attempts=3)
    for _ in range(3):
        with Session(eng) as s:
            run_all_cases(
                session=s,
                labels=labels,
                generator=fake,
                cases=("video",),
            )
    with Session(eng) as s:
        row = s.get(ClusterLabel, ("video", 0))
        assert row is not None
        assert row.status == "failed" and row.attempts == 3


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


def test_cluster_pass_audio_consumes_audio_clip_labels() -> None:
    """Stage-2 for ``case='audio'`` must read ``ClipLabel(label_case='audio')``
    rows, not the video-case rows. The prompt body the generator sees must
    embed the audio payload's distinctive substring.
    """
    eng = _engine()
    audio_marker = "speechy-podcast-clip-marker"
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(User(id=2, is_selected=True))
        for cid, uid in [(10, 1), (11, 1), (20, 2)]:
            s.add(Clip(id=cid, user_id=uid, is_selected=True, is_downloaded=True))
            # A visual row that MUST NOT leak into the audio cluster prompt.
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
            # The audio-case row that SHOULD appear in the audio prompt.
            s.add(
                ClipLabel(
                    clip_id=cid,
                    label_case="audio",
                    status="success",
                    validation="ok",
                    warnings=[],
                    payload={
                        "observable_audio_tags": [
                            {"tag": audio_marker, "evidence": "speech track"}
                        ],
                        "aesthetic_tags": [],
                        "community_signalling_tags": [],
                        "one_sentence_audio_reading": "spoken voice over",
                    },
                    attempts=1,
                )
            )
        for uid, centrality in [(1, 0.9), (2, 0.7)]:
            s.add(
                UserCluster(
                    user_id=uid,
                    embedding_case="audio",
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
            cases=("audio",),
        )

    assert len(fake.prompts) == 1, "expected exactly one cluster prompt for audio case"
    prompt = fake.prompts[0]
    assert audio_marker in prompt, (
        "audio cluster prompt must include the audio ClipLabel payload"
    )
    assert "visual-only-tag" not in prompt, (
        "audio cluster prompt must NOT include video-case clip labels"
    )

    with Session(eng) as s:
        row = s.get(ClusterLabel, ("audio", 0))
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
        assert row.payload["cluster_label"] == "soft domestic vignette"
        assert "marker" not in row.payload
    # Generator was actually invoked — proves the row was treated as
    # pending, not as a pre-existing success that the gate would seal.
    assert len(fake.calls) == 1
