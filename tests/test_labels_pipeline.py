import json
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy import event as _sqla_event
from sqlalchemy.orm import Session

from core.config import Secrets
from core.database import (
    Base,
    Clip,
    ClipLabel,
    ClusterLabel,
    User,
    UserCluster,
)
from core.database.engine import _apply_sqlite_pragmas
from modules.embeddings.cases import default_cases
from modules.labels.cases import REGISTRY
from tests._clustering_helpers import seed_audio_mir


@pytest.fixture
def db_engine(monkeypatch):
    eng = create_engine("sqlite:///:memory:")
    _sqla_event.listen(eng, "connect", _apply_sqlite_pragmas)
    Base.metadata.create_all(eng)
    import core.database.engine as engine_mod

    monkeypatch.setattr(engine_mod, "_main_engine", eng)
    yield eng
    eng.dispose()


def _settings(**label_overrides):
    from core.config import _load_settings

    s = _load_settings()
    for k, v in label_overrides.items():
        setattr(s.labels, k, v)
    return s


def _secrets() -> Secrets:
    return Secrets(
        database_url="sqlite:///:memory:",
        identity_db_url="sqlite:///:memory:",
        hiker_api_key="",
        huggingface_token="",
    )


def _seed(eng, *, n_selected: int) -> None:
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        for cid in range(1, n_selected + 1):
            s.add(Clip(id=cid, user_id=1, is_selected=True, is_downloaded=True))
        s.commit()


def _clean_json() -> str:
    return json.dumps(
        {
            "observable_visual_tags": [
                {"tag": "warm kitchen scene", "evidence": "lamp"},
                {"tag": "shallow depth field", "evidence": "blur"},
                {"tag": "handheld framing", "evidence": "drift"},
            ],
            "aesthetic_tags": [
                {
                    "tag": "soft domestic vignette",
                    "grounded_in": ["warm kitchen scene"],
                    "confidence": "medium",
                },
                {
                    "tag": "warm-toned framing",
                    "grounded_in": ["warm kitchen scene"],
                    "confidence": "low",
                },
                {
                    "tag": "intimate handheld register",
                    "grounded_in": ["handheld framing"],
                    "confidence": "high",
                },
            ],
            "community_signalling_tags": [
                {
                    "tag": "slow-living domestic taste",
                    "grounded_in": ["soft domestic vignette"],
                    "confidence": "low",
                },
                {
                    "tag": "homecore aesthetic register",
                    "grounded_in": ["warm-toned framing"],
                    "confidence": "medium",
                },
                {
                    "tag": "personal-diary reels register",
                    "grounded_in": ["intimate handheld register"],
                    "confidence": "low",
                },
            ],
            "one_sentence_visual_reading": "tight handheld kitchen vignette with warm domestic palette",
        }
    )


def _patch_generator(text_or_callable):
    """Patch ``LabelsGenerator.run`` (video case) and stub ``run_text``.

    Non-video cases either:
    - Reach ``run_text`` with whatever input the case adapter produced
      (sandwich, when a video ClipLabel exists), or
    - Short-circuit to ``status='failed'`` via the adapter returning
      ``None`` (audio/maest without speech/MIR; sandwich without a video
      label) — no ``run_text`` call in that branch.

    Returning the same response body for ``run_text`` is intentional: the
    validators reject case-shape mismatches (sandwich gets a video-shaped
    JSON ⇒ H1 failure), so non-video rows simply land as ``failed``
    without polluting the video-case assertions.
    """
    from modules.labels.models import LabelsGenerator

    def fake_run(self, video_path, prompt):
        if callable(text_or_callable):
            return text_or_callable(video_path, prompt)
        return text_or_callable

    def fake_run_many(self, video_paths, prompt):
        return [fake_run(self, vp, prompt) for vp in video_paths]

    def fake_run_text(self, prompt, *, max_new_tokens):
        if callable(text_or_callable):
            return text_or_callable(None, prompt)
        return text_or_callable

    return patch.multiple(
        LabelsGenerator,
        run=fake_run,
        run_many=fake_run_many,
        run_text=fake_run_text,
        unload=lambda self: None,
    )


def test_first_run_populates_rows(db_engine):
    _seed(db_engine, n_selected=3)
    s = _settings()
    with _patch_generator(_clean_json()):
        from modules.labels import run as run_labels

        run_labels(s, _secrets())
    with Session(db_engine) as sess:
        rows = (
            sess.query(ClipLabel)
            .filter(ClipLabel.label_case == "video")
            .order_by(ClipLabel.clip_id)
            .all()
        )
        assert [r.clip_id for r in rows] == [1, 2, 3]
        assert all(r.status == "success" for r in rows)
        assert all(r.validation == "ok" for r in rows)
        assert all(r.payload is not None for r in rows)


def test_idempotent_rerun_makes_no_model_calls(db_engine):
    _seed(db_engine, n_selected=2)
    # Pin sandwich + audio + maest to a single attempt so failures stay
    # terminal across the two runs (no_input / missing dependencies).
    s = _settings(max_attempts=1)
    video_calls = []
    text_calls = []

    def fake_run(self, video_path, prompt):
        video_calls.append(video_path)
        return _clean_json()

    def fake_run_many(self, video_paths, prompt):
        return [fake_run(self, vp, prompt) for vp in video_paths]

    def fake_run_text(self, prompt, *, max_new_tokens):
        text_calls.append(prompt)
        return _clean_json()  # video-shaped → validation failure for other cases

    from modules.labels.models import LabelsGenerator

    with patch.multiple(
        LabelsGenerator,
        run=fake_run,
        run_many=fake_run_many,
        run_text=fake_run_text,
        unload=lambda self: None,
    ):
        from modules.labels import run as run_labels

        run_labels(s, _secrets())
        first_video = list(video_calls)
        first_text = list(text_calls)
        run_labels(s, _secrets())
    # Second invocation must not issue any additional model calls.
    assert video_calls == first_video
    assert text_calls == first_text
    # Video case ran once per clip on the first call.
    assert len(first_video) == 2


def test_hard_fail_retries_until_max_attempts(db_engine):
    _seed(db_engine, n_selected=1)
    s = _settings(max_attempts=3)
    with _patch_generator("not json"):
        from modules.labels import run as run_labels

        for _ in range(3):
            run_labels(s, _secrets())
    with Session(db_engine) as sess:
        row = sess.get(ClipLabel, (1, "video"))
        assert row is not None
        assert row.attempts == 3
        assert row.status == "failed"
        assert row.error == "H1"
        # Other cases lack speech/MIR ⇒ adapter-level failure, never reach
        # validation. We only assert on the video case here.


def test_soft_fail_kept_as_warn(db_engine):
    _seed(db_engine, n_selected=1)
    s = _settings(min_tags_per_kind=10, max_attempts=1)
    with _patch_generator(_clean_json()):
        from modules.labels import run as run_labels

        run_labels(s, _secrets())
    with Session(db_engine) as sess:
        row = sess.get(ClipLabel, (1, "video"))
        assert row is not None
        assert row.status == "success"
        assert row.validation == "warn"
        assert "S1" in row.warnings


def test_prompt_drift_wipes_table(db_engine):
    _seed(db_engine, n_selected=1)
    s_a = _settings()
    s_b = _settings()
    s_b.labels.case_prompts = dict(s_b.labels.case_prompts)
    s_b.labels.case_prompts["video"] = s_b.labels.case_prompts["video"] + "\n# tweak"
    with _patch_generator(_clean_json()):
        from modules.labels import run as run_labels

        run_labels(s_a, _secrets())
        run_labels(s_b, _secrets())
    with Session(db_engine) as sess:
        rows = sess.query(ClipLabel).filter(ClipLabel.label_case == "video").all()
        assert len(rows) == 1
        assert rows[0].status == "success"


def test_selection_growth_adds_rows_without_wiping(db_engine):
    _seed(db_engine, n_selected=1)
    s = _settings()
    with _patch_generator(_clean_json()):
        from modules.labels import run as run_labels

        run_labels(s, _secrets())
    with Session(db_engine) as sess:
        sess.add(Clip(id=2, user_id=1, is_selected=True, is_downloaded=True))
        sess.commit()
    with _patch_generator(_clean_json()):
        from modules.labels import run as run_labels

        run_labels(s, _secrets())
    with Session(db_engine) as sess:
        ids = [
            r.clip_id
            for r in (
                sess.query(ClipLabel)
                .filter(ClipLabel.label_case == "video")
                .order_by(ClipLabel.clip_id)
            )
        ]
        assert ids == [1, 2]


def test_run_loops_all_default_cases_and_calls_run_text_for_non_video(db_engine):
    """``run`` dispatches the video case via ``run`` and non-video cases via
    ``run_text`` — the exact split depends on which adapters yield input.

    With a bare clip (no speech, no MIR), the non-video cases break down:
    - audio: no speech, no music ⇒ adapter returns ``None`` ⇒ no model call
    - sandwich: visual payload exists (video case ran first) ⇒ ``run_text``
    - maest: no MIR ⇒ adapter returns ``None`` ⇒ no model call
    """
    _seed(db_engine, n_selected=2)
    s = _settings(max_attempts=1)
    video_calls = []
    text_calls = []

    def fake_run(self, video_path, prompt):
        video_calls.append((str(video_path), prompt))
        return _clean_json()

    def fake_run_many(self, video_paths, prompt):
        return [fake_run(self, vp, prompt) for vp in video_paths]

    def fake_run_text(self, prompt, *, max_new_tokens):
        text_calls.append((prompt, max_new_tokens))
        return _clean_json()  # validates as video-shape only

    from modules.labels.models import LabelsGenerator

    with patch.multiple(
        LabelsGenerator,
        run=fake_run,
        run_many=fake_run_many,
        run_text=fake_run_text,
        unload=lambda self: None,
    ):
        from modules.labels import run as run_labels

        run_labels(s, _secrets())

    # One ``run`` (video) per selected clip.
    assert len(video_calls) == 2
    # Sandwich is the only non-video case where the adapter yields a
    # non-``None`` input under this minimal seed ⇒ exactly two ``run_text``
    # invocations, one per clip.
    assert len(text_calls) == 2
    # The sandwich prompts must NOT be the video prompt body.
    video_prompt = s.labels.case_prompts["video"]
    for prompt, _max in text_calls:
        assert not prompt.startswith(video_prompt + "\n\n")


# ---------------------------------------------------------------------------
# Phase F1 — end-to-end pipeline coverage
# ---------------------------------------------------------------------------

# Distinguishing tokens for the per-case clip stage-1 prompts. The case
# adapter prepends the case's prompt body (which contains the case's unique
# ``observable_*_tags`` key) before the input text, so the stub generator
# dispatches by scanning the prompt for that key — no call-order / sequence
# reasoning needed.
_CLIP_PROMPT_TOKENS: dict[str, str] = {
    "video": "observable_visual_tags",
    "audio": "observable_audio_tags",
    "sandwich": "observable_multimodal_tags",
    "maest": "observable_music_tags",
    "gemini": "observable_multimodal_tags",
}

# Cluster prompts mention ``dominant_*_repertoire`` keys uniquely per case.
_CLUSTER_PROMPT_TOKENS: dict[str, str] = {
    "video": "dominant_visual_repertoire",
    "audio": "dominant_audio_repertoire",
    "sandwich": "dominant_multimodal_repertoire",
    "maest": "dominant_music_repertoire",
    "gemini": "dominant_multimodal_repertoire",
}


def _clip_payload_for_case(case: str) -> dict:
    """Schema-valid stage-1 clip payload for an arbitrary case."""
    spec = REGISTRY[case]
    observable_key = next(
        k for k in spec.clip_required_keys if k.startswith("observable_")
    )
    sentence_key = next(
        k for k in spec.clip_required_keys if k.startswith("one_sentence_")
    )
    return {
        observable_key: [
            {"tag": f"{case} observation a", "evidence": "ev a"},
            {"tag": f"{case} observation b", "evidence": "ev b"},
            {"tag": f"{case} observation c", "evidence": "ev c"},
        ],
        "aesthetic_tags": [
            {
                "tag": f"{case} aesthetic a",
                "grounded_in": [f"{case} observation a"],
                "confidence": "medium",
            },
            {
                "tag": f"{case} aesthetic b",
                "grounded_in": [f"{case} observation b"],
                "confidence": "low",
            },
            {
                "tag": f"{case} aesthetic c",
                "grounded_in": [f"{case} observation c"],
                "confidence": "high",
            },
        ],
        "community_signalling_tags": [
            {
                "tag": f"{case} signalling a",
                "grounded_in": [f"{case} aesthetic a"],
                "confidence": "low",
            },
            {
                "tag": f"{case} signalling b",
                "grounded_in": [f"{case} aesthetic b"],
                "confidence": "medium",
            },
            {
                "tag": f"{case} signalling c",
                "grounded_in": [f"{case} aesthetic c"],
                "confidence": "low",
            },
        ],
        sentence_key: f"a {case} sentence reading the clip",
    }


def _cluster_payload_for_case(case: str) -> dict:
    """Schema-valid stage-2 cluster payload for an arbitrary case."""
    spec = REGISTRY[case]
    repertoire_key = next(
        k
        for k in spec.cluster_required_keys
        if k.startswith("dominant_") and k.endswith("_repertoire")
    )
    return {
        "cluster_label": f"{case} cluster label",
        "cluster_summary": f"{case} cluster summary",
        repertoire_key: [
            {
                "tag": f"{case} repertoire a",
                "description": "recurring strand a",
                "recurrence": "dominant",
            },
            {
                "tag": f"{case} repertoire b",
                "description": "recurring strand b",
                "recurrence": "frequent",
            },
        ],
        "dominant_aesthetic_logic": [
            {
                "tag": f"{case} logic",
                "grounded_in": [f"{case} repertoire a"],
                "description": "how the strands cohere",
            }
        ],
        "taste_signalling": {
            "label": f"{case} taste",
            "description": "taste read",
            "confidence": "medium",
        },
        "visibility_orientation": {
            "label": f"{case} visibility",
            "description": "visibility read",
            "confidence": "low",
        },
        "internal_variations": [
            {
                "variation": f"{case} variation",
                "description": "minor strand",
            }
        ],
        "boundary_notes": f"{case} differs from neighbours",
        "tool_tags": [f"{case}-tool"],
    }


class _DispatchingGen:
    """Stub generator dispatching on prompt content, not call order.

    The case-specific ``observable_*_tags`` (stage-1) and
    ``dominant_*_repertoire`` (stage-2) keys are present in the case's
    prompt body, so the stub inspects the prompt and returns the matching
    JSON payload.
    """

    def __init__(self) -> None:
        self.video_calls: list[tuple[str, str]] = []
        self.text_calls: list[tuple[str, int]] = []

    @staticmethod
    def _first_match(prompt: str, tokens: dict[str, str]) -> str | None:
        """Case whose token appears earliest in ``prompt`` (or None).

        The case-prompt body — which leads the assembled prompt — uniquely
        identifies the case; downstream embedded inputs (e.g. the sandwich
        case's ``VISUAL_OBSERVATIONS`` block carrying video keys) only
        ever appear AFTER the case-prompt prefix, so the earliest match
        wins.
        """
        best_case: str | None = None
        best_pos = -1
        for case, token in tokens.items():
            pos = prompt.find(token)
            if pos == -1:
                continue
            if best_pos == -1 or pos < best_pos:
                best_pos = pos
                best_case = case
        return best_case

    def _dispatch_clip(self, prompt: str) -> str:
        case = self._first_match(prompt, _CLIP_PROMPT_TOKENS)
        if case is None:
            raise AssertionError(f"unrecognised clip prompt: {prompt[:120]!r}")
        return json.dumps(_clip_payload_for_case(case))

    def _dispatch_cluster(self, prompt: str) -> str | None:
        case = self._first_match(prompt, _CLUSTER_PROMPT_TOKENS)
        if case is None:
            return None
        return json.dumps(_cluster_payload_for_case(case))

    def run(self, video_path, prompt):
        self.video_calls.append((str(video_path), prompt))
        return self._dispatch_clip(prompt)

    def run_text(self, prompt, *, max_new_tokens):
        self.text_calls.append((prompt, max_new_tokens))
        cluster = self._dispatch_cluster(prompt)
        if cluster is not None:
            return cluster
        return self._dispatch_clip(prompt)

    def unload(self) -> None:
        return None


def _patch_dispatching(gen: _DispatchingGen):
    from modules.labels.models import LabelsGenerator

    return patch.multiple(
        LabelsGenerator,
        run=lambda self, video_path, prompt: gen.run(video_path, prompt),
        run_many=lambda self, video_paths, prompt: [
            gen.run(vp, prompt) for vp in video_paths
        ],
        run_text=lambda self, prompt, *, max_new_tokens: gen.run_text(
            prompt, max_new_tokens=max_new_tokens
        ),
        unload=lambda self: None,
    )


def _seed_full_clip(eng, *, tmp_path) -> None:
    """Seed one user + one clip with speech + MIR + a fake video file."""
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(
            Clip(
                id=1,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                caption_clean="warm kitchen scene",
                caption_language="en",
                is_speech_detected=True,
                speech_transcription="a calm slow narration",
                speech_language="en",
            )
        )
        s.commit()
        seed_audio_mir(s, clip_id=1)
    # Materialise a tiny "video" file so ``file_stat_for_hash`` returns a
    # stable (size, mtime) tuple instead of the missing-file sentinel.
    (tmp_path / "1.mp4").write_bytes(b"\x00")


def _full_settings(tmp_path):
    s = _settings(max_attempts=1)
    s.paths.video_dir = str(tmp_path)
    return s


def test_full_run_writes_one_clip_label_per_case_per_clip(db_engine, tmp_path):
    _seed_full_clip(db_engine, tmp_path=tmp_path)
    settings = _full_settings(tmp_path)
    gen = _DispatchingGen()
    with _patch_dispatching(gen):
        from modules.labels import run as run_labels

        run_labels(settings, _secrets())

    expected_cases = set(default_cases(settings))
    with Session(db_engine) as sess:
        rows = sess.query(ClipLabel).filter(ClipLabel.clip_id == 1).all()
        cases_present = {r.label_case for r in rows}
        assert cases_present == expected_cases
        by_case = {r.label_case: r for r in rows}
        # With the full seed (speech + MIR + video file + video ClipLabel
        # from the video case running first) every default case has a
        # non-None adapter output → every row is ``status='success'``.
        for case in expected_cases:
            row = by_case[case]
            assert row.status == "success", (
                f"case={case} status={row.status} error={row.error!r}"
            )
            assert row.validation in ("ok", "warn")
            assert row.payload is not None


def test_full_run_writes_one_cluster_label_per_case(db_engine, tmp_path):
    _seed_full_clip(db_engine, tmp_path=tmp_path)
    settings = _full_settings(tmp_path)
    # Seed one UserCluster row per case so ``_load_candidates`` has a
    # cluster per case to label.
    with Session(db_engine) as sess:
        for case in default_cases(settings):
            sess.add(
                UserCluster(
                    user_id=1,
                    embedding_case=case,
                    cluster_id=0,
                    umap_x=0.0,
                    umap_y=0.0,
                    centrality=0.9,
                )
            )
        sess.commit()

    gen = _DispatchingGen()
    with _patch_dispatching(gen):
        from modules.labels import run as run_labels

        run_labels(settings, _secrets())

    expected_cases = set(default_cases(settings))
    with Session(db_engine) as sess:
        rows = sess.query(ClusterLabel).all()
        seen = {(r.embedding_case, r.cluster_id) for r in rows}
        for case in expected_cases:
            assert (case, 0) in seen, f"missing cluster label for case={case}"
        # Every per-case cluster row should be successful given the full
        # seed and the dispatching stub.
        by_case = {r.embedding_case: r for r in rows}
        for case in expected_cases:
            assert by_case[case].status == "success", (
                f"case={case} status={by_case[case].status} "
                f"error={by_case[case].error!r}"
            )


def test_idempotent_full_rerun(db_engine, tmp_path):
    _seed_full_clip(db_engine, tmp_path=tmp_path)
    settings = _full_settings(tmp_path)
    with Session(db_engine) as sess:
        for case in default_cases(settings):
            sess.add(
                UserCluster(
                    user_id=1,
                    embedding_case=case,
                    cluster_id=0,
                    umap_x=0.0,
                    umap_y=0.0,
                    centrality=0.9,
                )
            )
        sess.commit()

    gen = _DispatchingGen()
    with _patch_dispatching(gen):
        from modules.labels import run as run_labels

        run_labels(settings, _secrets())
        first_video = len(gen.video_calls)
        first_text = len(gen.text_calls)
        run_labels(settings, _secrets())
    # Second run must issue zero additional model calls.
    assert len(gen.video_calls) == first_video, (
        f"video calls grew: {first_video} → {len(gen.video_calls)}"
    )
    assert len(gen.text_calls) == first_text, (
        f"text calls grew: {first_text} → {len(gen.text_calls)}"
    )
