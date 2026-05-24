"""Pydantic mirrors of frontend/src/data/schemas/*.ts (Zod).

The frontend Zod schemas are the source of truth for the JSON contract;
these pydantic models reproduce that shape on the writer side so we
catch contract drift at write time, not at the browser. Any change to
the Zod schemas requires a matching change here, in the same PR.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal[1] = 1

EmbeddingCase = Literal["video", "sandwich", "audio"]


class BoundsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minX: float
    maxX: float
    minY: float
    maxY: float


class ClusterModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    label: str
    cx: float
    cy: float
    rx: float = Field(ge=0)
    ry: float = Field(ge=0)
    angle: float
    size: int = Field(ge=0)


class ManifestRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    case: EmbeddingCase
    label: str
    size: int = Field(ge=0)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = SCHEMA_VERSION
    default_run_id: str
    runs: list[ManifestRun] = Field(min_length=1)


class UsersFile(BaseModel):
    """Each user row is a 4-tuple: [id, x, y, cluster_id].

    cluster_id == -1 indicates a noise point.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = SCHEMA_VERSION
    run_id: str
    bounds: BoundsModel
    users: list[tuple[int, float, float, int]]


class ClustersFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = SCHEMA_VERSION
    run_id: str
    clusters: list[ClusterModel]
