import type { BulkSource } from "./bulk/BulkSource";
import { StaticBulkSource } from "./bulk/StaticBulkSource";
import { HttpBulkSource } from "./bulk/HttpBulkSource";
import type { ApiClient } from "./api/ApiClient";
import { StaticApiClient } from "./api/StaticApiClient";
import { HttpApiClient } from "./api/HttpApiClient";

export type SourcesConfig = {
  /** `import.meta.env.BASE_URL` — root for the static `data/` folder. */
  baseUrl: string;
  /** `VITE_API_BASE_URL` — when set, both planes load from the HTTP API. */
  apiBaseUrl?: string | undefined;
};

export type Sources = { bulk: BulkSource; api: ApiClient };

/**
 * Single switch for both data planes. With `apiBaseUrl` set, bulk + detail
 * both load from the read-only HTTP API; unset (the default), both load from
 * the static JSON tree — so the GitHub Pages deploy is unaffected.
 *
 * `getActiveRunId` resolves the active run on every detail call (matching the
 * static client) so the API client follows version switches without recreation.
 */
export function makeSources(
  config: SourcesConfig,
  getActiveRunId: () => string | null,
): Sources {
  if (config.apiBaseUrl) {
    return {
      bulk: new HttpBulkSource(config.apiBaseUrl),
      api: new HttpApiClient(config.apiBaseUrl, getActiveRunId),
    };
  }
  const dataRoot = `${config.baseUrl}data/`;
  return {
    bulk: new StaticBulkSource(dataRoot),
    api: new StaticApiClient(dataRoot, getActiveRunId),
  };
}
