"""Backend Pydantic mirrors validate the shipped version-6 fixtures.

These mirror the frontend Zod schemas (``frontend/src/data/schemas/*.ts``).
Parsing the actually-shipped JSON tree under ``frontend/public/data`` proves
the mirror matches the live contract, so the offload/reconstruct layers can
validate every payload before serialisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from modules.visualization.contract import (
    ClusterDetailModel,
    ClustersFileModel,
    CreatorDetailModel,
    ManifestModel,
    UsersFileModel,
)

_DATA = Path(__file__).resolve().parents[1] / "frontend" / "public" / "data"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


# The static JSON bundle is no longer shipped (the frontend is served from the
# atlas API). When a local export exists these mirrors still validate it;
# otherwise skip — the same contract models are enforced at the API reconstruct
# layer (see tests/test_atlas_api.py, tests/test_reconstruct.py).
if not (_DATA / "runs").is_dir():
    pytest.skip(
        "static data bundle absent (API-only); contract covered by atlas API tests",
        allow_module_level=True,
    )


# Derive the shipped run ids from the data tree so this tracks the live case
# set (e.g. auditory/sandwich/video) instead of a stale hardcoded list.
_RUN_IDS = sorted(
    p.name for p in (_DATA / "runs").iterdir() if (p / "users.json").exists()
)


def test_manifest_fixture_parses():
    m = ManifestModel.model_validate(_load(_DATA / "manifest.json"))
    assert m.version == 6
    assert m.runs


@pytest.mark.parametrize("run_id", _RUN_IDS)
def test_users_clusters_fixtures_parse(run_id):
    users = UsersFileModel.model_validate(_load(_DATA / "runs" / run_id / "users.json"))
    assert users.version == 6
    assert users.run_id == run_id
    # 6-tuple arity.
    assert all(len(u) == 6 for u in users.users)

    clusters = ClustersFileModel.model_validate(
        _load(_DATA / "runs" / run_id / "clusters.json")
    )
    assert clusters.version == 6


def test_all_creator_details_parse():
    user_dir = _DATA / "runs" / "video" / "users"
    files = sorted(user_dir.glob("*.json"))
    assert files
    for f in files:
        CreatorDetailModel.model_validate(_load(f))


def test_all_cluster_details_parse():
    cluster_dir = _DATA / "runs" / "video" / "clusters"
    files = sorted(cluster_dir.glob("*.json"))
    assert files
    for f in files:
        ClusterDetailModel.model_validate(_load(f))


def test_mirror_forbids_extra_keys():
    bad = _load(_DATA / "manifest.json")
    bad["unexpected"] = 1
    with pytest.raises(ValidationError):
        ManifestModel.model_validate(bad)
