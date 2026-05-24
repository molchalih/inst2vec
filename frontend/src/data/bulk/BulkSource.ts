import type { AtlasRun, Manifest } from "../types";

/**
 * The static-data plane. Surface is closed: only `getManifest` and
 * `getRun`. Per-creator details NEVER land here; they belong to ApiClient.
 */
export interface BulkSource {
  getManifest(): Promise<Manifest>;
  getRun(runId: string): Promise<AtlasRun>;
}
