import { describe, expect, it } from "vitest";
import { phraseFor } from "./phrase";

describe("phraseFor", () => {
  it.each([
    ["is_electronic", 2.44, { label: "electronic", arrow: "↑" }],
    ["is_electronic", -1.5, { label: "non-electronic", arrow: "↓" }],
    ["is_acoustic", 1.2, { label: "acoustic", arrow: "↑" }],
    ["danceability", 1.41, { label: "danceable", arrow: "↑" }],
    ["danceability", -0.9, { label: "undanceable", arrow: "↓" }],
    ["engagement", 1.0, { label: "engaging", arrow: "↑" }],
    ["approachability", -1.1, { label: "challenging", arrow: "↓" }],
    ["median_clip_duration_s", 1.03, { label: "longer clips", arrow: "↑" }],
    ["median_clip_duration_s", -1.0, { label: "shorter clips", arrow: "↓" }],
    ["median_clips_per_week", 1.5, { label: "posts often", arrow: "↑" }],
    ["engagement_shape_ratio", 1.2, { label: "viral shape", arrow: "↑" }],
    ["activity_span_months", 1.2, { label: "long history", arrow: "↑" }],
  ])("%s z=%f → %o", (feature, z, expected) => {
    const e = { feature, z, cohort_value: 0, baseline_mean: 0, baseline_std: 1 };
    expect(phraseFor(e)).toEqual(expected);
  });

  it("unknown feature → raw name with arrow", () => {
    const e = { feature: "mystery_metric", z: 2, cohort_value: 0, baseline_mean: 0, baseline_std: 1 };
    expect(phraseFor(e)).toEqual({ label: "mystery_metric", arrow: "↑" });
  });

  it("z=0 chooses positive label (no negative side rendered)", () => {
    const e = { feature: "danceability", z: 0, cohort_value: 0, baseline_mean: 0, baseline_std: 1 };
    expect(phraseFor(e).arrow).toBe("↑");
  });
});
