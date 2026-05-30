import { manifestSchema } from "../schemas/manifest.schema";
import { usersFileSchema } from "../schemas/users.schema";
import { clustersFileSchema } from "../schemas/clusters.schema";
import type { AtlasRun, Manifest } from "../types";
import type { BulkSource } from "./BulkSource";

/**
 * Bulk source backed by the read-only HTTP API (`services/atlas_api`).
 *
 * Identical to {@link StaticBulkSource} except the base is the API root rather
 * than the static `data/` folder; the API mirrors the static path suffixes
 * 1:1 (`manifest.json`, `runs/<id>/users.json`, …) and returns byte-identical
 * payloads, so the same Zod schemas parse both planes.
 */
export class HttpBulkSource implements BulkSource {
  /**
   * @param baseUrl Root URL of the API, e.g. `https://api.example.com/`.
   *   Must end with a trailing slash.
   */
  constructor(private readonly baseUrl: string) {
    if (!baseUrl.endsWith("/")) {
      throw new Error("HttpBulkSource baseUrl must end with '/'");
    }
  }

  async getManifest(): Promise<Manifest> {
    const raw = await this.fetchJson(`${this.baseUrl}manifest.json`);
    return manifestSchema.parse(raw);
  }

  async getRun(runId: string): Promise<AtlasRun> {
    const [usersRaw, clustersRaw, manifest] = await Promise.all([
      this.fetchJson(`${this.baseUrl}runs/${runId}/users.json`),
      this.fetchJson(`${this.baseUrl}runs/${runId}/clusters.json`),
      this.getManifest(),
    ]);
    const users = usersFileSchema.parse(usersRaw);
    const clusters = clustersFileSchema.parse(clustersRaw);
    const meta = manifest.runs.find((r) => r.id === runId);
    if (!meta) throw new Error(`Run ${runId} not found in manifest`);
    return {
      meta,
      bounds: users.bounds,
      users: users.users,
      clusters: clusters.clusters,
    };
  }

  private async fetchJson(url: string): Promise<unknown> {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Fetch ${url}: ${res.status}`);
    return res.json();
  }
}
