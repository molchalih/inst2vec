"""Tests for the global cluster-label naming pass (modules.labels.cluster_naming).

Covers the pure helpers (validity, lexical-overlap scoring, response parsing,
deterministic exact-uniqueness backstop, prompt/schema construction) and the
end-to-end orchestration through ``run_all_cases``: a well-behaved model yields
distinct length-consistent labels; a degenerate first round is repaired by the
feedback loop; a model that never cooperates still falls back to a clean,
non-truncating deterministic disambiguation.
"""

import json
from dataclasses import dataclass, field

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.config import LabelsSettings
from core.database import Base, Clip, ClipLabel, ClusterLabel, User, UserCluster
from modules.labels.cluster_naming import (
    BANNED_WORDS,
    MAX_SHARED_WORD,
    build_naming_prompt,
    candidate_phrases,
    content_words,
    naming_schema,
    overlap_report,
    parse_naming_response,
    phrase_to_label,
    resolve_lexical_overlap,
    valid_label,
)
from modules.labels.cluster_pass import run_all_cases
from modules.labels.validation import CLUSTER_LABEL_MAX_CHARS

# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def test_valid_label_accepts_title_case_two_to_three_words():
    assert valid_label("Cinematic Stillness")
    assert valid_label("Ambient Hip-Hop")
    assert valid_label("Avant-Garde Electronica")
    assert valid_label("Domestic Craft Rituals")


def test_valid_label_rejects_bad_shapes():
    assert not valid_label("")
    assert not valid_label("Single")  # one word
    assert not valid_label("lower case words")  # not title case
    assert not valid_label("Way Too Many Words Here")  # 5 words
    assert not valid_label("A" * (CLUSTER_LABEL_MAX_CHARS + 1))
    assert not valid_label(123)  # type
    assert not valid_label("Bad Under_score")  # underscore not allowed


def test_valid_label_rejects_banned_filler():
    assert "aesthetic" in BANNED_WORDS
    assert not valid_label("Domestic Aesthetic")
    assert not valid_label("Curated Stillness")


def test_content_words_splits_hyphens_and_drops_stopwords():
    assert content_words("Ambient Hip-Hop") == {"ambient", "hip", "hop"}
    # stopwords + short tokens removed
    assert content_words("Songs Of A City") == {"songs", "city"}


def test_overlap_report_flags_duplicates_and_overuse():
    labels = {
        0: "Ambient Tension",
        1: "Ambient Fusion",
        2: "Ambient Intimacy",
        3: "Urban Grooves",
    }
    rep = overlap_report(labels)
    assert rep.max_word_freq == 3  # "ambient" in 0,1,2
    assert "ambient" in rep.overused(max_shared=2)
    assert not rep.is_clean(max_shared=2)
    assert "ambient" in rep.feedback(max_shared=2).lower()


def test_overlap_report_clean_when_distinct():
    labels = {0: "Cinematic Tension", 1: "Urban Grooves", 2: "Analog Warmth"}
    rep = overlap_report(labels)
    assert rep.is_clean()
    assert rep.feedback() == ""


def test_overlap_report_detects_exact_duplicates():
    rep = overlap_report({0: "Same Name", 1: "Same Name", 2: "Other Thing"})
    assert rep.duplicates  # non-empty
    assert "DUPLICATE" in rep.feedback().upper()


def test_parse_naming_response_filters_invalid_and_unexpected():
    raw = json.dumps(
        [
            {"cluster_id": 0, "label": "Cinematic Tension"},
            {"cluster_id": 1, "label": "lower bad"},  # invalid label
            {"cluster_id": 9, "label": "Outside Set"},  # unexpected id
            {"cluster_id": 2, "label": "Urban Grooves"},
        ]
    )
    out = parse_naming_response(raw, expected_cids={0, 1, 2})
    assert out == {0: "Cinematic Tension", 2: "Urban Grooves"}


def test_parse_naming_response_unwraps_object_and_curly_quotes():
    raw = '{"labels": [{"cluster_id": 0, "label": "Analog Warmth"}]}'
    assert parse_naming_response(raw, {0}) == {0: "Analog Warmth"}
    # first-occurrence wins on duplicate ids
    raw2 = json.dumps(
        [
            {"cluster_id": 0, "label": "First Pick"},
            {"cluster_id": 0, "label": "Second Pick"},
        ]
    )
    assert parse_naming_response(raw2, {0}) == {0: "First Pick"}


def test_resolve_lexical_overlap_breaks_exact_collisions_from_own_elements():
    # Every model label is identical; the resolver must split them using each
    # cluster's OWN distinctive phrases, never a generic suffix or id.
    model = {0: "Shared Name", 1: "Shared Name", 2: "Shared Name"}
    cands = {
        0: ["Warm Kitchen"],
        1: ["Neon Strobe"],
        2: ["Analog Grain"],
    }
    out = resolve_lexical_overlap(model, cands)
    assert len({v.lower() for v in out.values()}) == 3
    assert all(valid_label(v) for v in out.values())
    rep = overlap_report(out)
    assert rep.max_word_freq <= MAX_SHARED_WORD


def test_resolve_lexical_overlap_fallback_stays_valid_under_pressure():
    # No candidate phrases and identical model labels: the deep fallback must
    # still produce distinct, VALID (2-3 word Title-Case) labels, never a
    # 4-word / numeric-suffix escape, when the cluster has usable words.
    model = {i: "Ambient Texture" for i in range(4)}
    cands = {
        0: ["Deep House", "Vaporwave Drift"],
        1: ["Grime Dubstep", "Bass Weight"],
        2: ["Swing Revival", "Euro Pulse"],
        3: ["Boom Bap", "Gypsy Jazz"],
    }
    out = resolve_lexical_overlap(model, cands)
    assert len({v.lower() for v in out.values()}) == 4
    assert all(valid_label(v) for v in out.values())
    assert overlap_report(out).max_word_freq <= MAX_SHARED_WORD


def test_phrase_to_label_builds_valid_titlecase_labels():
    assert phrase_to_label("ethereal minimalism") == "Ethereal Minimalism"
    assert phrase_to_label("synth-driven aggression") == "Synth Driven Aggression"
    assert phrase_to_label("cinematic-industrial-epic") == "Cinematic Industrial Epic"
    # >3 words: keep the leading 3
    assert phrase_to_label("hip hop jazzy hip hop") == "Hip Hop Jazzy"
    # one usable word -> cannot make a 2-word label
    assert phrase_to_label("synthesizer") is None
    assert phrase_to_label("the of a") is None
    # banned word -> rejected by valid_label
    assert phrase_to_label("domestic aesthetic") is None


def test_candidate_phrases_prefers_aesthetic_then_repertoire():
    payload = {
        "dominant_aesthetic_logic": [
            {"tag": "ethereal minimalism"},
            {"tag": "club-ready groove"},
        ],
        "dominant_music_repertoire": [
            {"tag": "electronic vaporwave"},
            {"tag": "synthesizer"},  # single usable word -> dropped
        ],
    }
    cands = candidate_phrases(payload, "dominant_music_repertoire")
    assert cands[0] == "Ethereal Minimalism"
    assert "Club Ready Groove" in cands
    assert "Electronic Vaporwave" in cands
    assert all(valid_label(c) for c in cands)


def test_resolve_lexical_overlap_caps_word_frequency():
    # Every model label leans on "ambient"; candidates supply distinctive
    # phrases so the resolver caps "ambient" at MAX_SHARED_WORD.
    model = {i: "Ambient Texture" for i in range(5)}
    cands = {
        0: ["Cinematic Tension", "Dark Synth"],
        1: ["Ethereal Minimalism", "Deep House"],
        2: ["Vaporwave Groove", "Glitchy Warmth"],
        3: ["Melodic Trap", "Boom Bap"],
        4: ["Swing Revival", "Euro House"],
    }
    out = resolve_lexical_overlap(model, cands)
    rep = overlap_report(out)
    assert rep.max_word_freq <= MAX_SHARED_WORD
    assert len({v.lower() for v in out.values()}) == 5  # all distinct
    assert all(valid_label(v) for v in out.values())


def test_resolve_lexical_overlap_rebuilds_banned_label():
    model = {0: "Domestic Aesthetic", 1: "Urban Realism"}  # 0 is banned
    cands = {0: ["Analog Craft", "Warm Kitchen"], 1: []}
    out = resolve_lexical_overlap(model, cands)
    assert valid_label(out[0]) and "aesthetic" not in out[0].lower()
    assert out[1] == "Urban Realism"


def test_naming_schema_constrains_label_pattern():
    schema = naming_schema()
    assert schema["type"] == "array"
    label_schema = schema["items"]["properties"]["label"]
    assert label_schema["maxLength"] == CLUSTER_LABEL_MAX_CHARS
    assert "pattern" in label_schema
    assert schema["items"]["properties"]["cluster_id"]["type"] == "integer"


def test_build_naming_prompt_includes_roster_rules_and_feedback():
    from modules.labels.cluster_naming import RosterEntry

    roster = [
        RosterEntry(0, "warm domestic kitchen scenes", ["warm kitchen", "handheld"]),
        RosterEntry(1, "neon club strobe energy", ["neon strobe"]),
    ]
    prompt = build_naming_prompt(
        "INSTRUCTIONS",
        roster,
        {0: "Working Zero", 1: "Working One"},
        feedback="OVERUSED words: x.",
    )
    assert "INSTRUCTIONS" in prompt
    assert "[0]" in prompt and "[1]" in prompt
    assert "warm kitchen" in prompt
    assert "aesthetic" in prompt  # banned-words listing
    assert "OVERUSED words" in prompt
    assert "cluster_id" in prompt  # output contract


# --------------------------------------------------------------------------- #
# Integration through run_all_cases                                            #
# --------------------------------------------------------------------------- #


def _labels(**overrides) -> LabelsSettings:
    base = dict(
        case_prompts={"video": "x"},
        cluster_case_prompts={"video": "cluster prompt"},
    )
    base.update(overrides)
    return LabelsSettings(**base)


def _clean_cluster_json(label: str, tags: list[str]) -> dict:
    rep = [
        {
            "tag": t,
            "description": "a recurring element across clips",
            "recurrence": "frequent",
        }
        for t in tags
    ] or [{"tag": "thing", "description": "x" * 30, "recurrence": "frequent"}]
    return {
        "cluster_label": label,
        "cluster_summary": "x" * 40
        + f" summary mentioning {tags[0] if tags else 'thing'}",
        "dominant_visual_repertoire": rep,
        "dominant_aesthetic_logic": [
            {
                "tag": "logic one",
                "grounded_in": [rep[0]["tag"]],
                "description": "y" * 30,
            }
        ],
        "taste_signalling": {
            "label": "t",
            "description": "z" * 30,
            "confidence": "medium",
        },
        "visibility_orientation": {
            "label": "v",
            "description": "w" * 30,
            "confidence": "low",
        },
        "internal_variations": [],
        "boundary_notes": "differs from neighbours in some grounded way here",
        "tool_tags": ["alpha", "beta", "gamma"],
    }


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


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


@dataclass
class _ScriptedGen:
    """Per-cluster calls return canned cluster JSON (round-robin); the single
    naming call per round returns the next scripted naming-array string."""

    cluster_payloads: list[str]
    naming_payloads: list[str]
    naming_prompts: list[str] = field(default_factory=list)
    _cluster_i: int = 0
    _naming_i: int = 0

    def _is_naming(self, prompt: str) -> bool:
        return "cluster_id" in prompt and "key elements" in prompt

    def run_text(self, prompt, **kw):
        if self._is_naming(prompt):
            self.naming_prompts.append(prompt)
            out = self.naming_payloads[
                min(self._naming_i, len(self.naming_payloads) - 1)
            ]
            self._naming_i += 1
            return out
        out = self.cluster_payloads[self._cluster_i % len(self.cluster_payloads)]
        self._cluster_i += 1
        return out

    def run_text_batch(self, prompts, *, seeds, **kw):
        return [self.run_text(p, **kw) for p in prompts]

    def reclaim_memory(self):  # pragma: no cover
        pass


def test_naming_pass_applies_distinct_model_labels():
    eng = _engine()
    _seed_n_clusters(eng, 3)
    cluster_payloads = [
        json.dumps(_clean_cluster_json("Generic Name", ["warm kitchen"])),
        json.dumps(_clean_cluster_json("Generic Name", ["neon strobe"])),
        json.dumps(_clean_cluster_json("Generic Name", ["analog grain"])),
    ]
    naming = json.dumps(
        [
            {"cluster_id": 0, "label": "Warm Kitchen"},
            {"cluster_id": 1, "label": "Neon Strobe"},
            {"cluster_id": 2, "label": "Analog Grain"},
        ]
    )
    gen = _ScriptedGen(cluster_payloads=cluster_payloads, naming_payloads=[naming])
    with Session(eng) as s:
        run_all_cases(session=s, labels=_labels(), generator=gen, cases=("video",))
    with Session(eng) as s:
        got = {
            cid: s.get(ClusterLabel, ("video", cid)).payload["cluster_label"]
            for cid in range(3)
        }
    assert got == {0: "Warm Kitchen", 1: "Neon Strobe", 2: "Analog Grain"}
    assert all(valid_label(v) for v in got.values())
    assert len(gen.naming_prompts) == 1  # converged in one round


def test_naming_pass_retries_on_lexical_overlap_then_converges():
    eng = _engine()
    _seed_n_clusters(eng, 3)
    cluster_payloads = [
        json.dumps(_clean_cluster_json("Generic Name", ["warm kitchen"]))
    ]
    overlapping = json.dumps(
        [
            {"cluster_id": 0, "label": "Ambient Tension"},
            {"cluster_id": 1, "label": "Ambient Fusion"},
            {"cluster_id": 2, "label": "Ambient Drift"},
        ]
    )
    fixed = json.dumps(
        [
            {"cluster_id": 0, "label": "Ambient Tension"},
            {"cluster_id": 1, "label": "Urban Fusion"},
            {"cluster_id": 2, "label": "Analog Drift"},
        ]
    )
    gen = _ScriptedGen(
        cluster_payloads=cluster_payloads,
        naming_payloads=[overlapping, fixed],
    )
    with Session(eng) as s:
        run_all_cases(
            session=s,
            labels=_labels(cluster_dedup_max_rounds=3),
            generator=gen,
            cases=("video",),
        )
    with Session(eng) as s:
        got = {
            cid: s.get(ClusterLabel, ("video", cid)).payload["cluster_label"]
            for cid in range(3)
        }
    # Second round's de-overlapped set wins; "ambient" no longer over-used.
    rep = overlap_report(got)
    assert rep.is_clean()
    assert len(gen.naming_prompts) == 2  # one overlap round + one converged round
    # the retry prompt must carry the overlap feedback
    assert "OVERUSED" in gen.naming_prompts[1]


def test_naming_pass_falls_back_deterministically_when_model_useless():
    eng = _engine()
    _seed_n_clusters(eng, 3)
    # Per-cluster pass emits the same label; naming model returns garbage every
    # round → deterministic exact-uniqueness backstop must still split them.
    cluster_payloads = [
        json.dumps(_clean_cluster_json("Shared Name", ["warm kitchen"])),
        json.dumps(_clean_cluster_json("Shared Name", ["neon strobe"])),
        json.dumps(_clean_cluster_json("Shared Name", ["analog grain"])),
    ]
    gen = _ScriptedGen(cluster_payloads=cluster_payloads, naming_payloads=["not json"])
    with Session(eng) as s:
        run_all_cases(
            session=s,
            labels=_labels(cluster_dedup_max_rounds=2),
            generator=gen,
            cases=("video",),
        )
    with Session(eng) as s:
        labels = [
            s.get(ClusterLabel, ("video", cid)).payload["cluster_label"]
            for cid in range(3)
        ]
    assert len({lab.lower() for lab in labels}) == 3, labels
    assert all(len(lab) <= CLUSTER_LABEL_MAX_CHARS for lab in labels)
