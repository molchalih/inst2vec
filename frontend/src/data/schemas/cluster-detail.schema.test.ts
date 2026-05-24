import { describe, expect, it } from "vitest";
import { clusterDetailSchema } from "./cluster-detail.schema";

const sample = {
  version: 2,
  cluster_id: 7,
  label: "Cluster 8",
  size: 42,
  ellipse: { cx: 1.23, cy: -0.45, rx: 0.82, ry: 0.31, angle: 0.61 },
  audio: { approachability: 0.61, engagement: 0.74, danceability: 0.58 },
  mood_shares: { happy: 0.4, sad: 0.05, relaxed: 0.3, aggressive: 0.1, party: 0.55 },
  timbre_shares: {
    acoustic: 0.2, electronic: 0.65, instrumental: 0.15,
    female_voice: 0.42, bright: 0.55, tonal: 0.78,
  },
  genre_top: [
    { label: "house", weight: 1.0 },
    { label: "techno", weight: 0.62 },
    { label: "pop", weight: 0.41 },
  ],
  instrument_top: [
    { label: "synth", weight: 1.0 },
    { label: "drums", weight: 0.83 },
  ],
  speech: {
    detected_share: 0.32,
    top_langs: [{ code: "en", share: 0.65 }, { code: "es", share: 0.2 }],
  },
  caption: {
    top_langs: [{ code: "en", share: 0.7 }, { code: "pt", share: 0.15 }],
  },
  posting: {
    median_plays: 18412,
    median_clip_duration_s: 24.3,
    median_clips_per_week: 4.2,
    engagement_shape_ratio: 1.8,
  },
  follower_bucket: "10k–20k",
  activity_span_months: 18,
  distinctiveness: [
    { feature: "is_electronic", cohort_value: 0.65, baseline_mean: 0.21, baseline_std: 0.18, z: 2.44 },
    { feature: "danceability", cohort_value: 0.58, baseline_mean: 0.41, baseline_std: 0.12, z: 1.41 },
    { feature: "median_clip_duration_s", cohort_value: 24.3, baseline_mean: 18.1, baseline_std: 6.0, z: 1.03 },
  ],
  spatial: {
    compactness: 0.019,
    nearest_clusters: [
      { cluster_id: 3, label: "Cluster 4", distance: 0.41 },
      { cluster_id: 12, label: "Cluster 13", distance: 0.58 },
      { cluster_id: 1, label: "Cluster 2", distance: 0.73 },
    ],
  },
};

describe("clusterDetailSchema", () => {
  it("parses the canonical sample payload", () => {
    const parsed = clusterDetailSchema.parse(sample);
    expect(parsed.cluster_id).toBe(7);
    expect(parsed.distinctiveness).toHaveLength(3);
    expect(parsed.spatial.nearest_clusters[0]!.cluster_id).toBe(3);
  });

  it("rejects unknown top-level fields", () => {
    const r = clusterDetailSchema.safeParse({ ...sample, surprise: 1 });
    expect(r.success).toBe(false);
  });

  it("rejects a v1 payload (version mismatch)", () => {
    const r = clusterDetailSchema.safeParse({ ...sample, version: 1 });
    expect(r.success).toBe(false);
  });
});
