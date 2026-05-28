"""One-shot dev helper: populate frontend/public/data with test label data.

What it does:
1. Walks `frontend/public/data/**/*.json` and bumps every prior-version
   payload (``"version": 5``) to ``TARGET_VERSION`` (6) so the frontend
   (now on SCHEMA_VERSION=6) can load the pre-existing static fixture set.
2. For every case under `frontend/public/data/runs/`, picks the first
   ``USERS_PER_CASE`` users with ``has_detail=true`` from that case's
   ``users.json`` and writes a synthesized creator-detail JSON to
   ``runs/{case}/users/{id}.json`` with rich ``clips: [...]`` payloads.
3. For every cluster detail JSON under ``runs/{case}/clusters/*.json``,
   bumps the ``version`` and writes a modality-appropriate ``label`` block
   replacing the old top-level ``label`` string. Per case:
   ``video`` → modality ``visual``, ``audio`` → ``audio``,
   ``sandwich`` → ``multimodal``, ``maest`` → ``music``.

Synthesized clip labels rotate through five aesthetic themes so the frontend
``SectionClips`` renderer exercises all three tag-kind chips, both ``ok`` and
``warn`` validation states, and the warning-string mapping. Numeric audio /
mood / posting fields are stable, plausible demo values — not real signal.

Run from the repo root:

    uv run python tests/fixtures/seed_test_labels.py

Idempotent. Re-running overwrites the synthesized creator JSONs and is a
no-op on already-bumped version fields.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path("frontend/public/data")
USERS_PER_CASE = None  # None = every eligible creator
TARGET_VERSION = 6
PREV_VERSION = 5

# Case → modality for the new ``label`` block. Mirrors
# ``modules.labels.cases.REGISTRY[case].modality`` without importing it
# (this script must run as a tiny standalone with no DB / settings).
CASE_MODALITY: dict[str, str] = {
    "video": "visual",
    "audio": "audio",
    "sandwich": "multimodal",
    "maest": "music",
}


# Five thematic clip-label sets. Each entry is a list of clip payloads
# matching the published per-user JSON shape (see
# frontend/src/data/schemas/clip-label.schema.ts).
THEMES: dict[str, list[dict]] = {
    "warm-domestic": [
        {
            "shortcode": "C_kitchen_1",
            "sentence": "tight handheld kitchen vignette with warm domestic palette and shallow depth of field",
            "tags": {
                "observable": [
                    {
                        "tag": "warm kitchen interior",
                        "evidence": "tungsten desk lamp on counter",
                    },
                    {
                        "tag": "shallow depth of field",
                        "evidence": "blurred background utensils",
                    },
                    {
                        "tag": "handheld camera drift",
                        "evidence": "subtle frame sway each cut",
                    },
                    {
                        "tag": "tea mug centre frame",
                        "evidence": "steaming mug held at chest height",
                    },
                ],
                "aesthetic": [
                    {
                        "tag": "soft domestic vignette",
                        "grounded_in": [
                            "warm kitchen interior",
                            "shallow depth of field",
                        ],
                        "confidence": "high",
                    },
                    {
                        "tag": "intimate handheld realism",
                        "grounded_in": ["handheld camera drift"],
                        "confidence": "medium",
                    },
                    {
                        "tag": "warm-toned framing",
                        "grounded_in": ["warm kitchen interior"],
                        "confidence": "high",
                    },
                ],
                "community": [
                    {
                        "tag": "slow-living domestic taste",
                        "grounded_in": ["soft domestic vignette"],
                        "confidence": "medium",
                    },
                    {
                        "tag": "homecore visual register",
                        "grounded_in": ["warm-toned framing", "warm kitchen interior"],
                        "confidence": "low",
                    },
                    {
                        "tag": "personal-diary reels register",
                        "grounded_in": ["intimate handheld realism"],
                        "confidence": "medium",
                    },
                ],
            },
            "validation": "ok",
            "warnings": [],
        },
        {
            "shortcode": "C_kitchen_2",
            "sentence": "morning routine sequence cut to soft acoustic music with pastel kitchen tones",
            "tags": {
                "observable": [
                    {
                        "tag": "pastel ceramic tableware",
                        "evidence": "cream and pink mugs",
                    },
                    {
                        "tag": "morning window light",
                        "evidence": "diffuse side light through curtain",
                    },
                    {
                        "tag": "soft acoustic underscore",
                        "evidence": "fingerpicked guitar audio",
                    },
                ],
                "aesthetic": [
                    {
                        "tag": "muted pastel palette",
                        "grounded_in": ["pastel ceramic tableware"],
                        "confidence": "high",
                    },
                    {
                        "tag": "gentle morning languor",
                        "grounded_in": [
                            "morning window light",
                            "soft acoustic underscore",
                        ],
                        "confidence": "medium",
                    },
                ],
                "community": [
                    {
                        "tag": "soft-living lifestyle register",
                        "grounded_in": ["gentle morning languor"],
                        "confidence": "low",
                    },
                    {
                        "tag": "kinfolk-adjacent aesthetic",
                        "grounded_in": ["muted pastel palette"],
                        "confidence": "medium",
                    },
                ],
            },
            "validation": "warn",
            "warnings": ["tag count out of range"],
        },
        {
            "shortcode": "C_kitchen_3",
            "sentence": "evening cooking close-up with steam, knife sounds, and amber overhead light",
            "tags": {
                "observable": [
                    {
                        "tag": "amber overhead pendant",
                        "evidence": "warm circular highlight on countertop",
                    },
                    {
                        "tag": "knife-on-board foley",
                        "evidence": "rhythmic chopping audio",
                    },
                    {
                        "tag": "steam rising frame top",
                        "evidence": "wisps over open pot",
                    },
                    {
                        "tag": "close-up macro detail",
                        "evidence": "lens within 30cm of subject",
                    },
                ],
                "aesthetic": [
                    {
                        "tag": "tactile culinary intimacy",
                        "grounded_in": [
                            "close-up macro detail",
                            "knife-on-board foley",
                        ],
                        "confidence": "high",
                    },
                    {
                        "tag": "warm evening glow",
                        "grounded_in": ["amber overhead pendant"],
                        "confidence": "high",
                    },
                ],
                "community": [
                    {
                        "tag": "slow-cooking enthusiast register",
                        "grounded_in": ["tactile culinary intimacy"],
                        "confidence": "medium",
                    },
                    {
                        "tag": "comfort-food domestic taste",
                        "grounded_in": ["warm evening glow"],
                        "confidence": "low",
                    },
                ],
            },
            "validation": "ok",
            "warnings": [],
        },
    ],
    "urban-minimal": [
        {
            "shortcode": "C_urban_1",
            "sentence": "wide-angle walking pov through monochrome modern city street at golden hour",
            "tags": {
                "observable": [
                    {
                        "tag": "wide-angle pov walking",
                        "evidence": "first-person framing chest height",
                    },
                    {
                        "tag": "monochrome concrete facades",
                        "evidence": "neutral grey buildings line frame",
                    },
                    {
                        "tag": "golden hour side light",
                        "evidence": "long shadow across pavement",
                    },
                    {
                        "tag": "steady gimbal motion",
                        "evidence": "level horizon no roll",
                    },
                ],
                "aesthetic": [
                    {
                        "tag": "clean architectural minimalism",
                        "grounded_in": ["monochrome concrete facades"],
                        "confidence": "high",
                    },
                    {
                        "tag": "cinematic urban stroll",
                        "grounded_in": [
                            "wide-angle pov walking",
                            "steady gimbal motion",
                        ],
                        "confidence": "high",
                    },
                ],
                "community": [
                    {
                        "tag": "urban-flâneur visual register",
                        "grounded_in": ["cinematic urban stroll"],
                        "confidence": "medium",
                    },
                    {
                        "tag": "design-conscious city taste",
                        "grounded_in": ["clean architectural minimalism"],
                        "confidence": "medium",
                    },
                ],
            },
            "validation": "ok",
            "warnings": [],
        },
        {
            "shortcode": "C_urban_2",
            "sentence": "rooftop drone reveal over downtown skyline with synth pad",
            "tags": {
                "observable": [
                    {
                        "tag": "drone rising reveal",
                        "evidence": "vertical altitude gain",
                    },
                    {
                        "tag": "downtown high-rise skyline",
                        "evidence": "glass towers fill mid frame",
                    },
                    {
                        "tag": "synth pad underscore",
                        "evidence": "sustained ambient chord",
                    },
                ],
                "aesthetic": [
                    {
                        "tag": "ambient cinematic scale",
                        "grounded_in": ["drone rising reveal", "synth pad underscore"],
                        "confidence": "high",
                    },
                ],
                "community": [
                    {
                        "tag": "city-pride aesthetic register",
                        "grounded_in": ["ambient cinematic scale"],
                        "confidence": "low",
                    },
                ],
            },
            "validation": "ok",
            "warnings": [],
        },
        {
            "shortcode": "C_urban_3",
            "sentence": "neon storefront pan with reflective puddle and low-angle composition",
            "tags": {
                "observable": [
                    {
                        "tag": "neon signage red and pink",
                        "evidence": "high-saturation glow",
                    },
                    {
                        "tag": "reflective wet pavement",
                        "evidence": "mirrored highlights in puddle",
                    },
                    {
                        "tag": "low-angle composition",
                        "evidence": "lens just above ground",
                    },
                ],
                "aesthetic": [
                    {
                        "tag": "cyberpunk-tinged urban moodboard",
                        "grounded_in": [
                            "neon signage red and pink",
                            "reflective wet pavement",
                        ],
                        "confidence": "medium",
                    },
                ],
                "community": [
                    {
                        "tag": "after-dark city register",
                        "grounded_in": ["cyberpunk-tinged urban moodboard"],
                        "confidence": "low",
                    },
                ],
            },
            "validation": "ok",
            "warnings": [],
        },
    ],
    "nature-soft": [
        {
            "shortcode": "C_nature_1",
            "sentence": "slow handheld pan across dew-covered meadow at dawn",
            "tags": {
                "observable": [
                    {
                        "tag": "dew-covered grass blades",
                        "evidence": "specular highlights catch light",
                    },
                    {
                        "tag": "dawn cool blue cast",
                        "evidence": "low colour temperature",
                    },
                    {
                        "tag": "slow lateral pan",
                        "evidence": "horizontal frame movement",
                    },
                ],
                "aesthetic": [
                    {
                        "tag": "tranquil dawn stillness",
                        "grounded_in": ["dawn cool blue cast", "slow lateral pan"],
                        "confidence": "high",
                    },
                    {
                        "tag": "naturalist observational gaze",
                        "grounded_in": ["dew-covered grass blades"],
                        "confidence": "medium",
                    },
                ],
                "community": [
                    {
                        "tag": "slow-nature lifestyle register",
                        "grounded_in": ["tranquil dawn stillness"],
                        "confidence": "medium",
                    },
                ],
            },
            "validation": "ok",
            "warnings": [],
        },
        {
            "shortcode": "C_nature_2",
            "sentence": "forest path walk with diffuse green canopy light and birdsong audio",
            "tags": {
                "observable": [
                    {
                        "tag": "diffuse green canopy light",
                        "evidence": "even shadowless illumination",
                    },
                    {
                        "tag": "birdsong layered audio",
                        "evidence": "ambient bird calls throughout",
                    },
                    {
                        "tag": "boots-on-leaves foley",
                        "evidence": "rhythmic dry-leaf crunch",
                    },
                ],
                "aesthetic": [
                    {
                        "tag": "verdant immersive calm",
                        "grounded_in": [
                            "diffuse green canopy light",
                            "birdsong layered audio",
                        ],
                        "confidence": "high",
                    },
                ],
                "community": [
                    {
                        "tag": "forest-bathing wellness register",
                        "grounded_in": ["verdant immersive calm"],
                        "confidence": "medium",
                    },
                ],
            },
            "validation": "ok",
            "warnings": [],
        },
    ],
    "studio-product": [
        {
            "shortcode": "C_studio_1",
            "sentence": "rotating product turntable on white seamless with directional rim light",
            "tags": {
                "observable": [
                    {
                        "tag": "white seamless backdrop",
                        "evidence": "uniform white background",
                    },
                    {
                        "tag": "rotating turntable motion",
                        "evidence": "constant rotational speed",
                    },
                    {
                        "tag": "directional rim light",
                        "evidence": "edge highlight along product silhouette",
                    },
                ],
                "aesthetic": [
                    {
                        "tag": "clinical studio precision",
                        "grounded_in": [
                            "white seamless backdrop",
                            "directional rim light",
                        ],
                        "confidence": "high",
                    },
                ],
                "community": [
                    {
                        "tag": "d2c-brand visual register",
                        "grounded_in": ["clinical studio precision"],
                        "confidence": "medium",
                    },
                ],
            },
            "validation": "warn",
            "warnings": ["hashtag-like tag", "ungrounded tag reference"],
        },
        {
            "shortcode": "C_studio_2",
            "sentence": "flat-lay overhead composition of skincare bottles arranged on marble",
            "tags": {
                "observable": [
                    {
                        "tag": "overhead flat-lay angle",
                        "evidence": "camera directly above subject",
                    },
                    {
                        "tag": "marble surface",
                        "evidence": "veined white slab fills frame",
                    },
                    {
                        "tag": "skincare bottle cluster",
                        "evidence": "5 amber and white containers",
                    },
                ],
                "aesthetic": [
                    {
                        "tag": "editorial flat-lay polish",
                        "grounded_in": ["overhead flat-lay angle", "marble surface"],
                        "confidence": "high",
                    },
                ],
                "community": [
                    {
                        "tag": "clean-beauty aesthetic register",
                        "grounded_in": ["editorial flat-lay polish"],
                        "confidence": "medium",
                    },
                ],
            },
            "validation": "ok",
            "warnings": [],
        },
    ],
    "moody-nightlife": [
        {
            "shortcode": "C_night_1",
            "sentence": "low-light club interior with strobe accents and shoulder-jostle handheld framing",
            "tags": {
                "observable": [
                    {
                        "tag": "low-light club interior",
                        "evidence": "near-black shadows fill frame",
                    },
                    {
                        "tag": "strobe colour accents",
                        "evidence": "magenta and cyan pulses",
                    },
                    {
                        "tag": "shoulder-jostle handheld",
                        "evidence": "frequent frame impacts",
                    },
                ],
                "aesthetic": [
                    {
                        "tag": "kinetic late-night euphoria",
                        "grounded_in": [
                            "strobe colour accents",
                            "shoulder-jostle handheld",
                        ],
                        "confidence": "high",
                    },
                ],
                "community": [
                    {
                        "tag": "club-going nightlife register",
                        "grounded_in": ["kinetic late-night euphoria"],
                        "confidence": "medium",
                    },
                ],
            },
            "validation": "ok",
            "warnings": [],
        },
        {
            "shortcode": "C_night_2",
            "sentence": "after-hours street snack queue under sodium street lights",
            "tags": {
                "observable": [
                    {
                        "tag": "sodium street light cast",
                        "evidence": "warm orange illumination",
                    },
                    {
                        "tag": "street-snack vendor stall",
                        "evidence": "smoking griddle in foreground",
                    },
                    {"tag": "queue of patrons", "evidence": "people standing in line"},
                ],
                "aesthetic": [
                    {
                        "tag": "city after-hours intimacy",
                        "grounded_in": [
                            "sodium street light cast",
                            "street-snack vendor stall",
                        ],
                        "confidence": "high",
                    },
                ],
                "community": [
                    {
                        "tag": "urban late-night-eats register",
                        "grounded_in": ["city after-hours intimacy"],
                        "confidence": "medium",
                    },
                ],
            },
            "validation": "ok",
            "warnings": [],
        },
    ],
}

THEME_NAMES = list(THEMES.keys())


def _bump_versions(root: Path) -> int:
    """Walk every JSON under root and bump top-level ``version`` to TARGET_VERSION."""
    bumped = 0
    for path in root.rglob("*.json"):
        text = path.read_text()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("version") == PREV_VERSION:
            payload["version"] = TARGET_VERSION
            path.write_text(json.dumps(payload))
            bumped += 1
    return bumped


def _demo_label_block(*, modality: str, fallback: str) -> dict:
    """Build a short canned ``label`` block for one cluster.

    Modality drives ``modality`` only; the rest of the block reuses the
    same canned strings across cases. Goal is JSON parsing under the
    frontend's clusterLabelSchema, not narrative quality.
    """
    return {
        "label": fallback or f"Sample {modality} cluster",
        "summary": f"Synthesized {modality}-modality cluster summary.",
        "modality": modality,
        "repertoire": [
            {
                "tag": f"{modality} recurring motif",
                "description": f"a recurring {modality} element across members",
                "recurrence": "dominant",
            },
            {
                "tag": f"{modality} secondary motif",
                "description": f"a secondary {modality} element",
                "recurrence": "frequent",
            },
        ],
        "aesthetic_logic": [
            {
                "tag": f"{modality} reading",
                "grounded_in": [f"{modality} recurring motif"],
                "description": f"the {modality} repertoire reads coherent",
            },
        ],
        "taste_signalling": {
            "label": f"{modality}-taste",
            "description": f"creators in this cluster share {modality}-mediated taste",
            "confidence": "medium",
        },
        "visibility_orientation": {
            "label": "steady",
            "description": "low-spectacle, steady attention",
            "confidence": "low",
        },
        "internal_variations": [
            {
                "variation": "minor strand",
                "description": f"a minor sub-strand within the {modality} cluster",
            }
        ],
        "boundary_notes": (
            f"differs from adjacent clusters by its {modality} register"
        ),
        "tool_tags": [f"{modality}-core", "demo"],
        "validation": "ok",
        "warnings": [],
    }


def _rewrite_cluster_detail(path: Path, *, modality: str) -> bool:
    """In-place rewrite of a cluster-detail JSON to schema v6.

    Replaces the legacy top-level ``label: <string>`` with the new
    case-agnostic ``label: ClusterLabel`` block, and bumps the version
    if needed. Returns True if the file was modified.
    """
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False
    legacy_label = payload.get("label")
    fallback = legacy_label if isinstance(legacy_label, str) else ""
    payload["version"] = TARGET_VERSION
    payload["label"] = _demo_label_block(modality=modality, fallback=fallback)
    path.write_text(json.dumps(payload))
    return True


def _rewrite_case_cluster_details(case_dir: Path) -> int:
    """For every ``clusters/*.json`` under one case, write a v6 label block."""
    clusters_dir = case_dir / "clusters"
    if not clusters_dir.exists():
        return 0
    modality = CASE_MODALITY.get(case_dir.name, "visual")
    n = 0
    for cluster_json in sorted(clusters_dir.glob("*.json")):
        if _rewrite_cluster_detail(cluster_json, modality=modality):
            n += 1
    return n


def _demo_creator_detail(
    *,
    user_id: int,
    cluster_id: int,
    x: float,
    y: float,
    theme: str,
) -> dict:
    clips = THEMES[theme]
    n_clips = len(clips)
    return {
        "version": TARGET_VERSION,
        "user_id": user_id,
        "cluster_id": cluster_id,
        "x": x,
        "y": y,
        "n_clips": n_clips,
        "audio": {
            "approachability": 0.62,
            "engagement": 0.71,
            "danceability": 0.54,
        },
        "mood_shares": {
            "happy": 0.42,
            "sad": 0.08,
            "relaxed": 0.31,
            "aggressive": 0.05,
            "party": 0.14,
        },
        "timbre_shares": {
            "acoustic": 0.46,
            "electronic": 0.19,
            "instrumental": 0.22,
            "female_voice": 0.31,
            "bright": 0.55,
            "tonal": 0.71,
        },
        "genre_top": [
            {"label": "indie-folk", "weight": 0.42},
            {"label": "ambient", "weight": 0.21},
            {"label": "lo-fi", "weight": 0.18},
        ],
        "instrument_top": [
            {"label": "acoustic-guitar", "weight": 0.51},
            {"label": "vocals", "weight": 0.28},
        ],
        "speech": {
            "detected_share": 0.6,
            "top_langs": [
                {"code": "en", "share": 0.71},
                {"code": "es", "share": 0.18},
            ],
        },
        "caption": {
            "detected_share": 0.62,
            "top_langs": [
                {"code": "en", "share": 0.85},
            ],
        },
        "posting": {
            "median_plays": 12400,
            "median_clip_duration_s": 18.5,
            "median_clips_per_week": 3.2,
            "engagement_shape_ratio": 0.43,
        },
        "follower_bucket": "10k–100k",
        "activity_span_months": 14,
        "distinctiveness": [
            {
                "feature": "is_acoustic",
                "cohort_value": 0.46,
                "baseline_mean": 0.21,
                "baseline_std": 0.12,
                "z": 2.08,
            },
        ],
        "spatial": {
            "distance_from_centroid": 0.42,
            "distance_from_centroid_percentile": 0.55,
            "nearest_other_cluster": {
                "cluster_id": (cluster_id + 1) if cluster_id >= 0 else 0,
                "label": "neighbouring register",
                "distance": 1.85,
            },
        },
        "clips": [
            {
                "clip_id": user_id * 1000 + i,
                "shortcode": clip["shortcode"],
                "thumbnail_url": None,
                "sentence": clip["sentence"],
                "tags": clip["tags"],
                "validation": clip["validation"],
                "warnings": clip["warnings"],
            }
            for i, clip in enumerate(clips)
        ],
    }


def _seed_case(case_dir: Path, count: int | None) -> int:
    """Pick users with has_detail=true; write per-user JSON files.

    ``count=None`` writes a file for every eligible creator.
    """
    users_json = case_dir / "users.json"
    if not users_json.exists():
        return 0
    bulk = json.loads(users_json.read_text())
    eligible = [u for u in bulk["users"] if u[4] is True and u[3] >= 0]
    if count is not None:
        eligible = eligible[:count]
    users_dir = case_dir / "users"
    users_dir.mkdir(exist_ok=True)
    for i, (uid, x, y, cluster_id, _has_detail, _centrality) in enumerate(eligible):
        theme = THEME_NAMES[i % len(THEME_NAMES)]
        detail = _demo_creator_detail(
            user_id=uid,
            cluster_id=cluster_id,
            x=float(x),
            y=float(y),
            theme=theme,
        )
        (users_dir / f"{uid}.json").write_text(json.dumps(detail))
    return len(eligible)


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Static data dir not found: {DATA_DIR.resolve()}")

    bumped = _bump_versions(DATA_DIR)
    print(f"bumped version {PREV_VERSION} → {TARGET_VERSION} in {bumped} files")

    runs_dir = DATA_DIR / "runs"
    total_users = 0
    total_clusters = 0
    for case_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        n_users = _seed_case(case_dir, USERS_PER_CASE)
        n_clusters = _rewrite_case_cluster_details(case_dir)
        print(
            f"  {case_dir.name}: {n_users} per-user JSONs, "
            f"{n_clusters} per-cluster JSONs rewritten"
        )
        total_users += n_users
        total_clusters += n_clusters
    print(
        f"done. {total_users} creator-detail and {total_clusters} "
        f"cluster-detail JSONs across {len(list(runs_dir.iterdir()))} cases."
    )


if __name__ == "__main__":
    main()
