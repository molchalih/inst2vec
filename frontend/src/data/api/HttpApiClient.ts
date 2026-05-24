import type { ClusterDetail } from "../schemas/cluster-detail.schema";
import type { CreatorDetail } from "../schemas/creator-detail.schema";
import type { ApiClient, CreatorSummary } from "./ApiClient";
import { ApiUnavailableError } from "./ApiClient";

/**
 * Stub for the future FastAPI client. The swap point in
 * app/providers.tsx already chooses between Http and Static based on
 * config; the impl lands when the backend ships.
 */
export class HttpApiClient implements ApiClient {
  constructor(private readonly baseUrl: string) {
    if (!baseUrl.endsWith("/")) {
      throw new Error("HttpApiClient baseUrl must end with '/'");
    }
  }

  getClusterDetail(_id: number): Promise<ClusterDetail> {
    return Promise.reject(new ApiUnavailableError("getClusterDetail (HttpApiClient not implemented)"));
  }
  getCreatorDetail(_id: number): Promise<CreatorDetail> {
    return Promise.reject(new ApiUnavailableError("getCreatorDetail (HttpApiClient not implemented)"));
  }
  searchCreators(_query: string): Promise<CreatorSummary[]> {
    return Promise.reject(new ApiUnavailableError("searchCreators (HttpApiClient not implemented)"));
  }
  getEdges(_creatorId: number): Promise<number[]> {
    return Promise.reject(new ApiUnavailableError("getEdges (HttpApiClient not implemented)"));
  }
  getReels(_creatorId: number): Promise<unknown[]> {
    return Promise.reject(new ApiUnavailableError("getReels (HttpApiClient not implemented)"));
  }
}
