import { describe, expect, it } from "vitest";
import { creatorDetailSchema } from "./creator-detail.schema";

const sample = {
  version: 7,
  user_id: 12345,
  cluster_id: 7,
  x: 1.42,
  y: -0.31,
  n_clips: 9,
  audio: { approachability: 0.58, engagement: 0.71, danceability: 0.62 },
  mood_shares: { happy: 0.44, sad: 0, relaxed: 0.22, aggressive: 0.11, party: 0.66 },
  timbre_shares: {
    acoustic: 0.11, electronic: 0.77, instrumental: 0.22,
    female_voice: 0.55, bright: 0.66, tonal: 0.88,
  },
  genre_top: [{ label: "house", weight: 1.0 }, { label: "techno", weight: 0.55 }],
  instrument_top: [{ label: "synth", weight: 1.0 }],
  speech: { detected_share: 0.44, top_langs: [{ code: "en", share: 1.0 }] },
  caption: { top_langs: [{ code: "en", share: 0.88 }] },
  posting: {
    median_plays: 26100,
    median_clip_duration_s: 22.0,
    median_clips_per_week: 3.1,
    engagement_shape_ratio: 2.4,
  },
  follower_bucket: "20k–50k",
  activity_span_months: 14,
  distinctiveness: [
    { feature: "danceability", cohort_value: 0.62, baseline_mean: 0.58, baseline_std: 0.09, z: 0.44 },
    { feature: "engagement_shape_ratio", cohort_value: 2.4, baseline_mean: 1.8, baseline_std: 0.5, z: 1.2 },
  ],
  spatial: {
    distance_from_centroid: 0.34,
    distance_from_centroid_percentile: 78,
    nearest_other_cluster: { cluster_id: 3, label: "Cluster 4", distance: 0.41 },
  },
  clips: [],
};

describe("creatorDetailSchema", () => {
  it("parses the canonical sample payload", () => {
    const parsed = creatorDetailSchema.parse(sample);
    expect(parsed.user_id).toBe(12345);
    expect(parsed.spatial.nearest_other_cluster?.cluster_id).toBe(3);
  });

  it("accepts a null nearest_other_cluster (non-borderline creator)", () => {
    const parsed = creatorDetailSchema.parse({
      ...sample,
      spatial: { ...sample.spatial, nearest_other_cluster: null },
    });
    expect(parsed.spatial.nearest_other_cluster).toBeNull();
  });

  it("rejects unknown top-level fields", () => {
    const r = creatorDetailSchema.safeParse({ ...sample, leaked_handle: "x" });
    expect(r.success).toBe(false);
  });
});
