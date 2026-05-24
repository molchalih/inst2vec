import { manifestSchema } from "../schemas/manifest.schema";
import { usersFileSchema } from "../schemas/users.schema";
import { clustersFileSchema } from "../schemas/clusters.schema";
import type { AtlasRun, Manifest } from "../types";
import type { BulkSource } from "./BulkSource";

export class StaticBulkSource implements BulkSource {
  /**
   * @param baseUrl Root URL of the data folder, e.g. `${import.meta.env.BASE_URL}data/`.
   *   Must end with a trailing slash.
   */
  constructor(private readonly baseUrl: string) {
    if (!baseUrl.endsWith("/")) {
      throw new Error("StaticBulkSource baseUrl must end with '/'");
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
