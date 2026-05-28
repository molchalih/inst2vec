import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { StaticApiClient, AssetNotFoundError } from "./StaticApiClient";
import { ApiUnavailableError } from "./ApiClient";

const ok = (body: unknown) =>
  Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));

const notFound = () =>
  Promise.resolve(new Response("missing", { status: 404 }));

const clusterDetailFixture = {
  version: 6,
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
};

const creatorDetailFixture = {
  version: 6,
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
  spatial: {
    distance_from_centroid: 0,
    distance_from_centroid_percentile: 0,
    nearest_other_cluster: null,
  },
  clips: [],
};

describe("StaticApiClient", () => {
  let originalFetch: typeof globalThis.fetch;

  beforeEach(() => { originalFetch = globalThis.fetch; });
  afterEach(() => { globalThis.fetch = originalFetch; });

  it("requires baseUrl to end with /", () => {
    expect(() => new StaticApiClient("/data", () => "video-1")).toThrow();
  });

  it("getClusterDetail: fetches and validates", async () => {
    globalThis.fetch = vi.fn(() => ok(clusterDetailFixture)) as never;
    const c = new StaticApiClient("/data/", () => "video-1");
    const detail = await c.getClusterDetail(7);
    expect(detail.cluster_id).toBe(7);
    expect(globalThis.fetch).toHaveBeenCalledWith("/data/runs/video-1/clusters/7.json");
  });

  it("getCreatorDetail: fetches and validates", async () => {
    globalThis.fetch = vi.fn(() => ok(creatorDetailFixture)) as never;
    const c = new StaticApiClient("/data/", () => "video-1");
    const detail = await c.getCreatorDetail(42);
    expect(detail.user_id).toBe(42);
    expect(globalThis.fetch).toHaveBeenCalledWith("/data/runs/video-1/users/42.json");
  });

  it("throws AssetNotFoundError on 404 so callers can fall back gracefully", async () => {
    globalThis.fetch = vi.fn(notFound) as never;
    const c = new StaticApiClient("/data/", () => "video-1");
    await expect(c.getCreatorDetail(42)).rejects.toBeInstanceOf(AssetNotFoundError);
    await expect(c.getClusterDetail(7)).rejects.toBeInstanceOf(AssetNotFoundError);
  });

  it("live-only methods throw ApiUnavailableError", async () => {
    const c = new StaticApiClient("/data/", () => "video-1");
    await expect(c.searchCreators("q")).rejects.toBeInstanceOf(ApiUnavailableError);
    await expect(c.getEdges(1)).rejects.toBeInstanceOf(ApiUnavailableError);
    await expect(c.getReels(1)).rejects.toBeInstanceOf(ApiUnavailableError);
  });
});
