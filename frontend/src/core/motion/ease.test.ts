import { describe, expect, it } from "vitest";
import { easeOutCubic } from "./ease";

describe("easeOutCubic", () => {
  it("is 0 at t=0", () => {
    expect(easeOutCubic(0)).toBe(0);
  });

  it("is 1 at t=1", () => {
    expect(easeOutCubic(1)).toBe(1);
  });

  it("is past 0.5 at t=0.5 (ease-out shape)", () => {
    const v = easeOutCubic(0.5);
    expect(v).toBeGreaterThan(0.5);
    expect(v).toBeLessThan(1);
  });

  it("matches the cubic-bezier(0.22,1,0.36,1) curve at t=0.25 within tolerance", () => {
    // Reference value sampled from the same cubic-bezier curve.
    expect(easeOutCubic(0.25)).toBeCloseTo(0.578, 2);
  });

  it("clamps inputs outside [0, 1] to the closest endpoint", () => {
    expect(easeOutCubic(-0.5)).toBe(0);
    expect(easeOutCubic(1.5)).toBe(1);
  });
});
