import type { ClusterDetail, ClusterLabel } from "../schemas/cluster-detail.schema";
import type { CreatorDetail } from "../schemas/creator-detail.schema";

export type { ClusterDetail, ClusterLabel } from "../schemas/cluster-detail.schema";
export type { CreatorDetail } from "../schemas/creator-detail.schema";

export interface CreatorSummary {
  id: number;
  label: string;
}

/**
 * The single Postgres-shaped contract. Today it's served from static
 * JSON via StaticApiClient; the FastAPI deploy later swaps in
 * HttpApiClient with no other code changes.
 *
 * `getClustersDetail` (the eager per-run bundle of every cluster's main
 * detail), `getClusterLabel` (the deferred per-cluster tag block, null when a
 * cluster has none), and `getCreatorDetail` always have an answer (static or
 * live). `searchCreators`, `getEdges`, `getReels` are live-only —
 * StaticApiClient throws ApiUnavailableError for them.
 */
export interface ApiClient {
  getClustersDetail(): Promise<ClusterDetail[]>;
  getClusterLabel(id: number): Promise<ClusterLabel | null>;
  getCreatorDetail(id: number): Promise<CreatorDetail>;

  searchCreators(query: string): Promise<CreatorSummary[]>;
  getEdges(creatorId: number): Promise<number[]>;
  getReels(creatorId: number): Promise<unknown[]>;
}

export class ApiUnavailableError extends Error {
  constructor(method: string) {
    super(`ApiClient.${method}: not available in this build`);
    this.name = "ApiUnavailableError";
  }
}
