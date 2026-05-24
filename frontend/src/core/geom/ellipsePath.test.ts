import { describe, expect, it } from "vitest";
import { ellipsePoints } from "./ellipsePath";

describe("ellipsePoints", () => {
  it("returns the requested number of points", () => {
    const pts = ellipsePoints({ cx: 0, cy: 0, rx: 1, ry: 1, angle: 0 }, 32);
    expect(pts).toHaveLength(32);
  });

  it("traces a unit circle when rx = ry = 1, angle = 0", () => {
    const pts = ellipsePoints({ cx: 0, cy: 0, rx: 1, ry: 1, angle: 0 }, 360);
    for (const p of pts) {
      const r = Math.hypot(p.x, p.y);
      expect(r).toBeCloseTo(1, 6);
    }
  });

  it("translates by (cx, cy)", () => {
    const pts = ellipsePoints({ cx: 5, cy: -2, rx: 1, ry: 1, angle: 0 }, 16);
    const meanX = pts.reduce((s, p) => s + p.x, 0) / pts.length;
    const meanY = pts.reduce((s, p) => s + p.y, 0) / pts.length;
    expect(meanX).toBeCloseTo(5, 6);
    expect(meanY).toBeCloseTo(-2, 6);
  });

  it("respects rotation: angle = π/2 swaps rx and ry semantics", () => {
    const pts = ellipsePoints({ cx: 0, cy: 0, rx: 2, ry: 1, angle: Math.PI / 2 }, 4);
    // First point at t=0 is at (rx, 0) un-rotated, which after rotating
    // 90° lands at (0, rx) = (0, 2).
    expect(pts[0]!.x).toBeCloseTo(0, 6);
    expect(pts[0]!.y).toBeCloseTo(2, 6);
  });

  it("clamps segment count to at least 3", () => {
    const pts = ellipsePoints({ cx: 0, cy: 0, rx: 1, ry: 1, angle: 0 }, 1);
    expect(pts.length).toBeGreaterThanOrEqual(3);
  });
});
