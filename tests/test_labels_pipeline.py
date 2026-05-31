import contextlib
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


def _fake_prepare_many(self, video_paths, prompt):
    """Test-shim: short-circuits the CPU prep half of the labels generator.

    Returns the inputs verbatim so the paired ``_fake_generate_from_inputs``
    can dispatch through ``self.run_many`` (the patch site's source of
    truth), bypassing real frame decoding + tokenization in tests.
    """
    return (list(video_paths), prompt)


def _fake_generate_from_inputs(self, inputs):
    video_paths, prompt = inputs
    return self.run_many(video_paths, prompt)


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

    def fake_run_text(
        self,
        prompt,
        *,
        max_new_tokens,
        seed=None,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        schema=None,
    ):
        if callable(text_or_callable):
            return text_or_callable(None, prompt)
        return text_or_callable

    return patch.multiple(
        LabelsGenerator,
        run=fake_run,
        run_many=fake_run_many,
        prepare_many=_fake_prepare_many,
        generate_from_inputs=_fake_generate_from_inputs,
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

    def fake_run_text(
        self,
        prompt,
        *,
        max_new_tokens,
        seed=None,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        schema=None,
    ):
        text_calls.append(prompt)
        return _clean_json()  # video-shaped → validation failure for other cases

    from modules.labels.models import LabelsGenerator

    with patch.multiple(
        LabelsGenerator,
        run=fake_run,
        run_many=fake_run_many,
        prepare_many=_fake_prepare_many,
        generate_from_inputs=_fake_generate_from_inputs,
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


def test_hard_validation_fail_goes_terminal_on_first_attempt(db_engine):
    """Validation hard-fails (H1/H2/H3) are deterministic under the seeded
    generator and short-circuit the retry budget — one attempt and the row
    is terminally ``failed``. Subsequent pipeline runs see no pending rows
    and issue no further model calls.
    """
    _seed(db_engine, n_selected=1)
    s = _settings(max_attempts=3)
    with _patch_generator("not json"):
        from modules.labels import run as run_labels

        # Three runs: only the first should reach the model — the H1
        # terminal write removes the row from ``_pending_clip_ids`` after
        # one attempt.
        for _ in range(3):
            run_labels(s, _secrets())
    with Session(db_engine) as sess:
        row = sess.get(ClipLabel, (1, "video"))
        assert row is not None
        assert row.status == "failed"
        assert row.attempts == s.labels.max_attempts
        assert row.error.startswith("H1 ")


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


def test_run_runs_stage1_for_video_only_and_skips_non_video_clip_pass(db_engine):
    """Only the video case runs the per-clip stage-1 pass.

    Non-video cases (sandwich/audio/maest) are stage-1-skipped by spec —
    their cluster pass synthesises from raw signals directly. With no
    ``UserCluster`` rows seeded, the cluster pass for every case finds
    no candidates and never calls ``run_text``. The only model calls in
    this run are the video case's stage-1 ``run`` invocations.
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

    def fake_run_text(
        self,
        prompt,
        *,
        max_new_tokens,
        seed=None,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        schema=None,
    ):
        text_calls.append((prompt, max_new_tokens))
        return _clean_json()  # validates as video-shape only

    from modules.labels.models import LabelsGenerator

    with patch.multiple(
        LabelsGenerator,
        run=fake_run,
        run_many=fake_run_many,
        prepare_many=_fake_prepare_many,
        generate_from_inputs=_fake_generate_from_inputs,
        run_text=fake_run_text,
        unload=lambda self: None,
    ):
        from modules.labels import run as run_labels

        run_labels(s, _secrets())

    # One ``run`` (video stage 1) per selected clip; non-video stage 1 is
    # skipped by spec, and the cluster pass has no UserCluster rows to
    # sample, so ``run_text`` is never reached.
    assert len(video_calls) == 2
    assert text_calls == []


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
    "spoken": "observable_audio_tags",
    "sandwich": "observable_multimodal_tags",
    "auditory": "observable_music_tags",
    "textual": "observable_textual_tags",
}

# Cluster prompts mention ``dominant_*_repertoire`` keys uniquely per case.
_CLUSTER_PROMPT_TOKENS: dict[str, str] = {
    "video": "dominant_visual_repertoire",
    "spoken": "dominant_audio_repertoire",
    "sandwich": "dominant_multimodal_repertoire",
    "auditory": "dominant_music_repertoire",
    "textual": "dominant_textual_repertoire",
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
    # Tags use "{case}-" prefix so the stub can dispatch by prompt content;
    # no connector words (with/for/and/of/to/in/on/the/a/an/&), ≤ 3 words,
    # ≤ 28 chars — HC5-clean.
    tag_one = f"{case}-primary"
    tag_two = f"{case}-secondary"
    return {
        "cluster_label": f"{case} cluster label",
        "cluster_summary": f"{case} cluster summary",
        repertoire_key: [
            {
                "tag": tag_one,
                "description": "recurring strand primary",
                "recurrence": "dominant",
            },
            {
                "tag": tag_two,
                "description": "recurring strand secondary",
                "recurrence": "frequent",
            },
        ],
        "dominant_aesthetic_logic": [
            {
                "tag": f"{case}-logic",
                "grounded_in": [tag_one],
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

    def run_text(
        self,
        prompt,
        *,
        max_new_tokens,
        seed=None,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        schema=None,
    ):
        self.text_calls.append((prompt, max_new_tokens))
        cluster = self._dispatch_cluster(prompt)
        if cluster is not None:
            return cluster
        return self._dispatch_clip(prompt)

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
    ):
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

    def unload(self) -> None:
        return None


@contextlib.contextmanager
def _patch_dispatching(gen: _DispatchingGen):
    from modules.labels.models import ClusterLabelsGenerator, LabelsGenerator

    def _run_text(
        self,
        prompt,
        *,
        max_new_tokens,
        seed=None,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        schema=None,
    ):
        return gen.run_text(prompt, max_new_tokens=max_new_tokens)

    def _run_text_batch(
        self,
        prompts,
        *,
        max_new_tokens,
        seeds,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        schema=None,
    ):
        return gen.run_text_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            seeds=seeds,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            schema=schema,
        )

    with (
        patch.multiple(
            LabelsGenerator,
            run=lambda self, video_path, prompt: gen.run(video_path, prompt),
            run_many=lambda self, video_paths, prompt: [
                gen.run(vp, prompt) for vp in video_paths
            ],
            prepare_many=_fake_prepare_many,
            generate_from_inputs=_fake_generate_from_inputs,
            run_text=_run_text,
            unload=lambda self: None,
        ),
        patch.multiple(
            ClusterLabelsGenerator,
            run_text_batch=_run_text_batch,
            unload=lambda self: None,
            reclaim_memory=lambda self: None,
        ),
    ):
        yield


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


def test_full_run_writes_clip_labels_only_for_stage1_backed_cases(db_engine, tmp_path):
    """Stage 1 only runs for cases with ``spec.runs_clip_pass=True``.

    Currently only the video case opts in — sandwich/audio/maest skip
    stage 1 and let the cluster pass synthesise from raw signals — so a
    full pipeline run writes exactly one ``ClipLabel`` row per selected
    clip per stage-1-backed case.
    """
    _seed_full_clip(db_engine, tmp_path=tmp_path)
    settings = _full_settings(tmp_path)
    gen = _DispatchingGen()
    with _patch_dispatching(gen):
        from modules.labels import run as run_labels

        run_labels(settings, _secrets())

    expected_cases = {c for c in default_cases(settings) if REGISTRY[c].runs_clip_pass}
    with Session(db_engine) as sess:
        rows = sess.query(ClipLabel).filter(ClipLabel.clip_id == 1).all()
        cases_present = {r.label_case for r in rows}
        assert cases_present == expected_cases
        for row in rows:
            assert row.status == "success", (
                f"case={row.label_case} status={row.status} error={row.error!r}"
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


def test_vl_unloads_before_cluster_generator_loads(db_engine, monkeypatch):
    """The VL-8B generator must be unloaded before the 30B cluster generator
    is constructed, and the cluster pass must run on the cluster generator.
    """
    from modules.labels import pipeline as pl

    events: list[str] = []

    class _FakeVL:
        def unload(self):
            events.append("vl_unload")

    class _FakeCluster:
        def unload(self):
            events.append("cluster_unload")

    monkeypatch.setattr(
        pl.LabelsGenerator, "lazy", classmethod(lambda cls, labels: _FakeVL())
    )
    monkeypatch.setattr(
        pl.ClusterLabelsGenerator,
        "lazy",
        classmethod(lambda cls, labels: _FakeCluster()),
    )
    monkeypatch.setattr(pl, "purge_orphans", lambda session: None)
    monkeypatch.setattr(pl.clip_pass, "run_case", lambda **kw: events.append("clip"))
    monkeypatch.setattr(
        pl.cluster_pass, "run_all_cases", lambda **kw: events.append("cluster_pass")
    )

    pl.run(_settings(), _secrets())

    assert "cluster_pass" in events
    assert events.index("clip") < events.index("vl_unload")
    # First VL unload happens before the cluster pass runs.
    assert events.index("vl_unload") < events.index("cluster_pass")
    # Cluster generator is unloaded only after the cluster pass.
    assert events.index("cluster_pass") < events.index("cluster_unload")
