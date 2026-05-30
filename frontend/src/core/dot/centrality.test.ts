import { describe, expect, it } from "vitest";
import { centralityRadiusScale, type CentralityScaleParams } from "./centrality";

const params: CentralityScaleParams = { min: 0.4, max: 2.4, gamma: 2 };

describe("centralityRadiusScale", () => {
  it("returns max at centrality=1", () => {
    expect(centralityRadiusScale(0, 1, params)).toBeCloseTo(params.max, 6);
  });

  it("returns min at centrality=0", () => {
    expect(centralityRadiusScale(0, 0, params)).toBeCloseTo(params.min, 6);
  });

  it("clamps centrality below 0 to the min", () => {
    expect(centralityRadiusScale(0, -1, params)).toBeCloseTo(params.min, 6);
  });

  it("clamps centrality above 1 to the max", () => {
    expect(centralityRadiusScale(0, 2, params)).toBeCloseTo(params.max, 6);
  });

  it("bypasses the curve for noise (clusterId < 0)", () => {
    expect(centralityRadiusScale(-1, 0.5, params)).toBe(1);
    expect(centralityRadiusScale(-1, 1, params)).toBe(1);
  });

  it("applies the gamma curve (convex pulls mid-centrality below linear midpoint)", () => {
    const linearMid = params.min + (params.max - params.min) * 0.5;
    const curved = centralityRadiusScale(0, 0.5, params);
    expect(curved).toBeLessThan(linearMid);
    // gamma=2 → 0.5^2 = 0.25, so curved = min + 0.25*(max-min)
    expect(curved).toBeCloseTo(params.min + (params.max - params.min) * 0.25, 6);
  });

  it("is monotonically non-decreasing in centrality for a signal cluster", () => {
    const a = centralityRadiusScale(0, 0.2, params);
    const b = centralityRadiusScale(0, 0.5, params);
    const c = centralityRadiusScale(0, 0.9, params);
    expect(a).toBeLessThan(b);
    expect(b).toBeLessThan(c);
  });
});
