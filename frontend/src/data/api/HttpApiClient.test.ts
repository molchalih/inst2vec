import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { HttpApiClient } from "./HttpApiClient";
import { AssetNotFoundError } from "./StaticApiClient";
import { ApiUnavailableError } from "./ApiClient";
import { SCHEMA_VERSION } from "../schemas/version";

const ok = (body: unknown) =>
  Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));
const notFound = () =>
  Promise.resolve(new Response("missing", { status: 404 }));

const mainDetail = {
  cluster_id: 7,
  size: 1,
  ellipse: { cx: 0, cy: 0, rx: 1, ry: 1, angle: 0 },
  audio: { approachability: 0.5, engagement: 0.5, danceability: 0.5 },
  mood_shares: { happy: 0, sad: 0, relaxed: 0, aggressive: 0, party: 0 },
  timbre_shares: { acoustic: 0, electronic: 0, instrumental: 0, female_voice: 0, bright: 0, tonal: 0 },
  genre_top: [],
  instrument_top: [],
  speech: { detected_share: 0, top_langs: [] },
  caption: { top_langs: [] },
  posting: { median_plays: 0, median_clip_duration_s: 0, median_clips_per_week: 0, engagement_shape_ratio: 0 },
  follower_bucket: "1k–2k",
  activity_span_months: 1,
  distinctiveness: [],
  spatial: { compactness: 0, nearest_clusters: [] },
  label_modality: null,
};

const clustersDetailBundle = {
  version: SCHEMA_VERSION,
  run_id: "video",
  clusters: [mainDetail],
};

const clusterLabelFile = {
  version: SCHEMA_VERSION,
  cluster_id: 7,
  label: {
    label: "soft domestic",
    summary: "kitchen scenes",
    modality: "visual",
    repertoire: [],
    aesthetic_logic: [],
    taste_signalling: { label: "homecore", description: "d", confidence: "medium" },
    visibility_orientation: { label: "ordinary", description: "d", confidence: "low" },
    internal_variations: [],
    boundary_notes: "",
    tool_tags: [],
    validation: "ok",
    warnings: [],
  },
};

const creatorDetailFixture = {
  version: SCHEMA_VERSION,
  user_id: 42,
  cluster_id: 7,
  x: 0, y: 0,
  n_clips: 1,
  audio: { approachability: 0.5, engagement: 0.5, danceability: 0.5 },
  mood_shares: { happy: 0, sad: 0, relaxed: 0, aggressive: 0, party: 0 },
  timbre_shares: { acoustic: 0, electronic: 0, instrumental: 0, female_voice: 0, bright: 0, tonal: 0 },
  genre_top: [],
  instrument_top: [],
  speech: { detected_share: 0, top_langs: [] },
  caption: { top_langs: [] },
  posting: { median_plays: 0, median_clip_duration_s: 0, median_clips_per_week: 0, engagement_shape_ratio: 0 },
  follower_bucket: "1k–2k",
  activity_span_months: 1,
  distinctiveness: [],
  spatial: { distance_from_centroid: 0, distance_from_centroid_percentile: 0, nearest_other_cluster: null },
  clips: [],
};

describe("HttpApiClient", () => {
  let originalFetch: typeof globalThis.fetch;
  beforeEach(() => { originalFetch = globalThis.fetch; });
  afterEach(() => { globalThis.fetch = originalFetch; });

  it("requires baseUrl to end with /", () => {
    expect(() => new HttpApiClient("https://api", () => "video")).toThrow();
  });

  it("getClustersDetail: GETs per-run clusters-detail.json and validates", async () => {
    globalThis.fetch = vi.fn(() => ok(clustersDetailBundle)) as never;
    const c = new HttpApiClient("https://api/", () => "video");
    const clusters = await c.getClustersDetail();
    expect(clusters[0]!.cluster_id).toBe(7);
    expect(globalThis.fetch).toHaveBeenCalledWith("https://api/runs/video/clusters-detail.json");
  });

  it("getClusterLabel: GETs per-cluster label file and validates", async () => {
    globalThis.fetch = vi.fn(() => ok(clusterLabelFile)) as never;
    const c = new HttpApiClient("https://api/", () => "video");
    const label = await c.getClusterLabel(7);
    expect(label?.modality).toBe("visual");
    expect(globalThis.fetch).toHaveBeenCalledWith("https://api/runs/video/clusters/7.label.json");
  });

  it("getClusterLabel: returns null on 404", async () => {
    globalThis.fetch = vi.fn(notFound) as never;
    const c = new HttpApiClient("https://api/", () => "video");
    expect(await c.getClusterLabel(7)).toBeNull();
  });

  it("getCreatorDetail: GETs per-run users/<id>.json and validates", async () => {
    globalThis.fetch = vi.fn(() => ok(creatorDetailFixture)) as never;
    const c = new HttpApiClient("https://api/", () => "video");
    const detail = await c.getCreatorDetail(42);
    expect(detail.user_id).toBe(42);
    expect(globalThis.fetch).toHaveBeenCalledWith("https://api/runs/video/users/42.json");
  });

  it("maps 404 to AssetNotFoundError for creator detail and the bundle", async () => {
    globalThis.fetch = vi.fn(notFound) as never;
    const c = new HttpApiClient("https://api/", () => "video");
    await expect(c.getCreatorDetail(42)).rejects.toBeInstanceOf(AssetNotFoundError);
    await expect(c.getClustersDetail()).rejects.toBeInstanceOf(AssetNotFoundError);
  });

  it("live-only methods reject with ApiUnavailableError", async () => {
    const c = new HttpApiClient("https://api/", () => "video");
    await expect(c.searchCreators("q")).rejects.toBeInstanceOf(ApiUnavailableError);
    await expect(c.getEdges(1)).rejects.toBeInstanceOf(ApiUnavailableError);
    await expect(c.getReels(1)).rejects.toBeInstanceOf(ApiUnavailableError);
  });
});
