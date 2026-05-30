import type { UsersFile } from "./schemas/users.schema";
import type { ClusterShape } from "./schemas/clusters.schema";
import type { ManifestRun } from "./schemas/manifest.schema";

export type AtlasRun = {
  meta: ManifestRun;
  bounds: UsersFile["bounds"];
  users: UsersFile["users"];
  clusters: ClusterShape[];
};

export type { ClusterShape } from "./schemas/clusters.schema";
export type { Manifest, ManifestRun, EmbeddingCase } from "./schemas/manifest.schema";
