import { describe, expect, it } from "vitest";
import {
  clusterDetailSchema,
  clustersDetailBundleSchema,
  clusterLabelFileSchema,
  type ClusterLabel,
} from "./cluster-detail.schema";
import { SCHEMA_VERSION } from "./version";

// One cluster's main detail — version-less and label-less, carrying only
// `label_modality`. This is the element shape inside `clusters-detail.json`.
const sample = {
  cluster_id: 7,
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
  label_modality: "visual",
};

const baseLabel = (): ClusterLabel => ({
  label: "soft domestic",
  summary: "tight handheld kitchen scenes",
  modality: "visual",
  repertoire: [
    { tag: "warm kitchen", description: "tungsten domestic rooms recurring", recurrence: "dominant" },
    { tag: "shallow focus", description: "blurred backgrounds across recurring shots", recurrence: "frequent" },
  ],
  aesthetic_logic: [
    { tag: "intimate realism", grounded_in: ["warm kitchen"], description: "warm handheld framing reads intimate not staged" },
  ],
  taste_signalling: { label: "homecore", description: "low-stakes domestic affinity expressed through repertoire", confidence: "medium" },
  visibility_orientation: { label: "ordinariness", description: "low spectacle low polish steady attention staging", confidence: "low" },
  internal_variations: [{ variation: "bathroom-lit", description: "minor strand of cool-lit grooming clips" }],
  boundary_notes: "differs from food-styling clusters by lacking top-down plating",
  tool_tags: ["homecore", "warm-palette", "handheld"],
  validation: "ok",
  warnings: [],
});

describe("clusterDetailSchema (main detail)", () => {
  it("parses the canonical sample payload", () => {
    const parsed = clusterDetailSchema.parse(sample);
    expect(parsed.cluster_id).toBe(7);
    expect(parsed.distinctiveness).toHaveLength(3);
    expect(parsed.spatial.nearest_clusters[0]!.cluster_id).toBe(3);
    expect(parsed.label_modality).toBe("visual");
  });

  it("accepts a null label_modality (cluster with no tags)", () => {
    const parsed = clusterDetailSchema.parse({ ...sample, label_modality: null });
    expect(parsed.label_modality).toBeNull();
  });

  it("rejects an unknown label_modality", () => {
    const r = clusterDetailSchema.safeParse({ ...sample, label_modality: "visualish" });
    expect(r.success).toBe(false);
  });

  it("rejects the heavy label block on the main detail", () => {
    const r = clusterDetailSchema.safeParse({ ...sample, label: baseLabel() });
    expect(r.success).toBe(false);
  });

  it("rejects unknown top-level fields", () => {
    const r = clusterDetailSchema.safeParse({ ...sample, surprise: 1 });
    expect(r.success).toBe(false);
  });
});

describe("clustersDetailBundleSchema", () => {
  it("parses a versioned per-run bundle of main details", () => {
    const parsed = clustersDetailBundleSchema.parse({
      version: SCHEMA_VERSION,
      run_id: "video",
      clusters: [sample, { ...sample, cluster_id: 8, label_modality: null }],
    });
    expect(parsed.clusters).toHaveLength(2);
    expect(parsed.run_id).toBe("video");
  });

  it("rejects a version mismatch", () => {
    const r = clustersDetailBundleSchema.safeParse({
      version: 1,
      run_id: "video",
      clusters: [sample],
    });
    expect(r.success).toBe(false);
  });
});

describe("clusterLabelFileSchema (deferred tags)", () => {
  it("parses a complete label file", () => {
    const parsed = clusterLabelFileSchema.parse({
      version: SCHEMA_VERSION,
      cluster_id: 7,
      label: baseLabel(),
    });
    expect(parsed.label.label).toBe("soft domestic");
    expect(parsed.label.modality).toBe("visual");
  });

  it("accepts warn-status label block with non-enum confidence", () => {
    const lbl = baseLabel();
    lbl.validation = "warn";
    lbl.warnings = ["invalid_confidence"];
    lbl.taste_signalling.confidence = "very high" as ClusterLabel["taste_signalling"]["confidence"];
    const r = clusterLabelFileSchema.safeParse({
      version: SCHEMA_VERSION,
      cluster_id: 7,
      label: lbl,
    });
    expect(r.success).toBe(true);
  });

  it("accepts a non-visual modality on the label block", () => {
    const r = clusterLabelFileSchema.parse({
      version: SCHEMA_VERSION,
      cluster_id: 7,
      label: { ...baseLabel(), modality: "audio" },
    });
    expect(r.label.modality).toBe("audio");
  });

  it("rejects an unknown modality on the label block", () => {
    const r = clusterLabelFileSchema.safeParse({
      version: SCHEMA_VERSION,
      cluster_id: 7,
      label: { ...baseLabel(), modality: "visualish" },
    });
    expect(r.success).toBe(false);
  });

  it("rejects a version mismatch", () => {
    const r = clusterLabelFileSchema.safeParse({
      version: 1,
      cluster_id: 7,
      label: baseLabel(),
    });
    expect(r.success).toBe(false);
  });
});
