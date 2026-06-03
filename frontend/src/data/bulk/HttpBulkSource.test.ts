import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { HttpBulkSource } from "./HttpBulkSource";

const ok = (body: unknown) =>
  Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));

const manifestFixture = {
  version: 7,
  default_run_id: "video",
  runs: [
    { id: "video", case: "video", label: "Visual", size: 2, details_available: true },
  ],
};

const usersFixture = {
  version: 7,
  run_id: "video",
  bounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 },
  users: [
    [0, 0.1, 0.2, 0, true, 0.5],
    [1, 0.3, 0.4, -1, false, 0],
  ],
};

const clustersFixture = {
  version: 7,
  run_id: "video",
  clusters: [
    { id: 0, label: "Cluster 1", cx: 0, cy: 0, rx: 1, ry: 1, angle: 0, size: 1, has_detail: true },
  ],
};

describe("HttpBulkSource", () => {
  let originalFetch: typeof globalThis.fetch;
  beforeEach(() => { originalFetch = globalThis.fetch; });
  afterEach(() => { globalThis.fetch = originalFetch; });

  it("requires baseUrl to end with /", () => {
    expect(() => new HttpBulkSource("https://api")).toThrow();
  });

  it("getManifest: GETs manifest.json and parses", async () => {
    globalThis.fetch = vi.fn(() => ok(manifestFixture)) as never;
    const src = new HttpBulkSource("https://api/");
    const m = await src.getManifest();
    expect(m.default_run_id).toBe("video");
    expect(globalThis.fetch).toHaveBeenCalledWith("https://api/manifest.json");
  });

  it("getRun: GETs users + clusters and returns AtlasRun", async () => {
    globalThis.fetch = vi.fn((url: string) => {
      if (url.endsWith("manifest.json")) return ok(manifestFixture);
      if (url.endsWith("users.json")) return ok(usersFixture);
      if (url.endsWith("clusters.json")) return ok(clustersFixture);
      throw new Error(`unexpected url ${url}`);
    }) as never;
    const src = new HttpBulkSource("https://api/");
    const run = await src.getRun("video");
    expect(run.meta.id).toBe("video");
    expect(run.users).toHaveLength(2);
    expect(run.clusters).toHaveLength(1);
    expect(globalThis.fetch).toHaveBeenCalledWith("https://api/runs/video/users.json");
    expect(globalThis.fetch).toHaveBeenCalledWith("https://api/runs/video/clusters.json");
  });

  it("getRun: throws when run absent from manifest", async () => {
    globalThis.fetch = vi.fn((url: string) => {
      if (url.endsWith("manifest.json")) return ok(manifestFixture);
      if (url.endsWith("users.json")) return ok({ ...usersFixture, run_id: "audio" });
      if (url.endsWith("clusters.json")) return ok({ ...clustersFixture, run_id: "audio" });
      throw new Error(`unexpected url ${url}`);
    }) as never;
    const src = new HttpBulkSource("https://api/");
    await expect(src.getRun("audio")).rejects.toThrow(/not found in manifest/);
  });
});
