import {
  clustersDetailBundleSchema,
  clusterLabelFileSchema,
} from "../schemas/cluster-detail.schema";
import { creatorDetailSchema } from "../schemas/creator-detail.schema";
import type { ClusterDetail, ClusterLabel } from "../schemas/cluster-detail.schema";
import type { CreatorDetail } from "../schemas/creator-detail.schema";
import type { ApiClient, CreatorSummary } from "./ApiClient";
import { ApiUnavailableError } from "./ApiClient";
import { AssetNotFoundError } from "./StaticApiClient";

/**
 * Live HTTP client for the read-only atlas API (`services/atlas_api`).
 *
 * Byte-for-byte identical to {@link StaticApiClient} except the base is the API
 * root rather than the static `data/` folder — the API mirrors the static path
 * suffixes 1:1, so the same Zod schemas parse both planes and a single
 * `VITE_API_BASE_URL` swap flips the data plane. The live-only methods
 * (`searchCreators`/`getEdges`/`getReels`) stay unimplemented, exactly like
 * the static client.
 */
export class HttpApiClient implements ApiClient {
  constructor(
    private readonly baseUrl: string,
    private readonly getActiveRunId: () => string | null,
  ) {
    if (!baseUrl.endsWith("/")) {
      throw new Error("HttpApiClient baseUrl must end with '/'");
    }
  }

  async getClustersDetail(): Promise<ClusterDetail[]> {
    const runId = this.requireRunId();
    const raw = await this.fetchJson(
      `${this.baseUrl}runs/${runId}/clusters-detail.json`,
    );
    return clustersDetailBundleSchema.parse(raw).clusters;
  }

  async getClusterLabel(id: number): Promise<ClusterLabel | null> {
    const runId = this.requireRunId();
    try {
      const raw = await this.fetchJson(
        `${this.baseUrl}runs/${runId}/clusters/${id}.label.json`,
      );
      return clusterLabelFileSchema.parse(raw).label;
    } catch (err) {
      if (err instanceof AssetNotFoundError) return null;
      throw err;
    }
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
    if (!id) throw new Error("HttpApiClient: no active runId");
    return id;
  }

  private async fetchJson(url: string): Promise<unknown> {
    const res = await fetch(url);
    if (res.status === 404) throw new AssetNotFoundError(url);
    if (!res.ok) throw new Error(`Fetch ${url}: ${res.status}`);
    return res.json();
  }
}
