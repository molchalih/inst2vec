import { describe, expect, it } from "vitest";
import { BruteForceHitTest } from "./hit-test";

describe("BruteForceHitTest", () => {
  const dots = [
    { id: 0, x: 0, y: 0, clusterId: 0 },
    { id: 1, x: 5, y: 5, clusterId: 0 },
    { id: 2, x: 5.2, y: 5.1, clusterId: 9 },
  ];
  const ht = new BruteForceHitTest(dots, []);

  it("finds the nearest dot within radius", () => {
    expect(ht.nearestDot({ x: 5.1, y: 5.05 }, 0.5)).toEqual({ id: 2, clusterId: 9 });
  });

  it("returns null when nothing is within radius", () => {
    expect(ht.nearestDot({ x: 100, y: 100 }, 0.5)).toBeNull();
  });

  it("falls back to ellipses when no dot matches", () => {
    const ellipses = [{ id: 7, cx: 10, cy: 10, rx: 2, ry: 1, angle: 0 }];
    const ht2 = new BruteForceHitTest(dots, ellipses);
    expect(ht2.ellipseAt({ x: 11, y: 10 })).toBe(7);
    expect(ht2.ellipseAt({ x: 20, y: 20 })).toBeNull();
  });

  it("prefers the smallest containing ellipse", () => {
    const ellipses = [
      { id: 1, cx: 0, cy: 0, rx: 1, ry: 1, angle: 0 },
      { id: 2, cx: 0, cy: 0, rx: 10, ry: 10, angle: 0 },
    ];
    const ht3 = new BruteForceHitTest([], ellipses);
    expect(ht3.ellipseAt({ x: 0, y: 0 })).toBe(1);
  });

  it("nearestDot returns the dot's cluster id alongside its id", () => {
    const localDots = [{ id: 7, x: 0, y: 0, clusterId: 3 }];
    const hit = new BruteForceHitTest(localDots, []);
    expect(hit.nearestDot({ x: 0, y: 0 }, 1)).toEqual({ id: 7, clusterId: 3 });
  });
});
