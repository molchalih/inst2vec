import { clusterDetailSchema } from "../schemas/cluster-detail.schema";
import { creatorDetailSchema } from "../schemas/creator-detail.schema";
import type { ClusterDetail } from "../schemas/cluster-detail.schema";
import type { CreatorDetail } from "../schemas/creator-detail.schema";
import { type ApiClient, type CreatorSummary, ApiUnavailableError } from "./ApiClient";

/**
 * Thrown when a static per-id JSON is missing (HTTP 404). Distinct
 * from generic fetch errors so callers can fall back gracefully —
 * e.g. creator-detail loads route to the cluster pane instead of
 * surfacing an error UI when the per-creator JSON was never exported.
 */
export class AssetNotFoundError extends Error {
  constructor(url: string) {
    super(`Asset not found: ${url}`);
    this.name = "AssetNotFoundError";
  }
}

/**
 * Drop-in for the future HttpApiClient. Reads static per-id JSON files
 * shipped alongside the bulk payload. Throws ApiUnavailableError for
 * methods that genuinely need a live backend.
 *
 * The active run id is read on every call via `getActiveRunId` so the
 * client follows version switches without being recreated.
 */
export class StaticApiClient implements ApiClient {
  constructor(
    private readonly baseUrl: string,
    private readonly getActiveRunId: () => string | null,
  ) {
    if (!baseUrl.endsWith("/")) {
      throw new Error("StaticApiClient baseUrl must end with '/'");
    }
  }

  async getClusterDetail(id: number): Promise<ClusterDetail> {
    const runId = this.requireRunId();
    const raw = await this.fetchJson(`${this.baseUrl}runs/${runId}/clusters/${id}.json`);
    return clusterDetailSchema.parse(raw);
  }

  async getCreatorDetail(id: number): Promise<CreatorDetail> {
    const runId = this.requireRunId();
    const raw = await this.fetchJson(`${this.baseUrl}runs/${runId}/users/${id}.json`);
    return creatorDetailSchema.parse(raw);
  }

  searchCreators(_query: string): Promise<CreatorSummary[]> {
    return Promise.reject(new ApiUnavailableError("searchCreators"));
  }

  getEdges(_creatorId: number): Promise<number[]> {
    return Promise.reject(new ApiUnavailableError("getEdges"));
  }

  getReels(_creatorId: number): Promise<unknown[]> {
    return Promise.reject(new ApiUnavailableError("getReels"));
  }

  private requireRunId(): string {
    const id = this.getActiveRunId();
    if (!id) throw new Error("StaticApiClient: no active runId");
    return id;
  }

  private async fetchJson(url: string): Promise<unknown> {
    const res = await fetch(url);
    if (res.status === 404) throw new AssetNotFoundError(url);
    if (!res.ok) throw new Error(`Fetch ${url}: ${res.status}`);
    return res.json();
  }
}
