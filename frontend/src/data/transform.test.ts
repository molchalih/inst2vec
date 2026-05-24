import { describe, expect, it } from "vitest";
import type { AtlasRun, ClusterShape } from "./types";
import { stretchRun } from "./transform";

const makeRun = (
  bounds: AtlasRun["bounds"],
  users: AtlasRun["users"],
  clusters: ClusterShape[],
): AtlasRun => ({
  meta: { id: "video-1", case: "video", label: "Visual", size: users.length, details_available: false },
  bounds,
  users,
  clusters,
});

describe("stretchRun", () => {
  it("returns the run unchanged when viewport width or height is zero or negative", () => {
    const run = makeRun({ minX: 0, maxX: 1, minY: 0, maxY: 1 }, [], []);
    expect(stretchRun(run, 0, 100)).toBe(run);
    expect(stretchRun(run, 100, 0)).toBe(run);
    expect(stretchRun(run, -1, 100)).toBe(run);
    expect(stretchRun(run, 100, -1)).toBe(run);
  });

  it("produces finite output when raw bounds collapse along one axis", () => {
    const run = makeRun(
      { minX: 5, maxX: 5, minY: 0, maxY: 10 },
      [[0, 5, 5, 0, false]],
      [{ id: 0, label: "thin", cx: 5, cy: 5, rx: 0, ry: 1, angle: 0, size: 1, has_detail: false }],
    );
    const out = stretchRun(run, 200, 200);
    const [, x, y] = out.users[0]!;
    expect(Number.isFinite(x)).toBe(true);
    expect(Number.isFinite(y)).toBe(true);
    expect(Number.isFinite(out.clusters[0]!.cx)).toBe(true);
    expect(Number.isFinite(out.clusters[0]!.angle)).toBe(true);
  });

  it("centers stretched bounds on the origin and matches the viewport size", () => {
    const run = makeRun(
      { minX: -1, maxX: 1, minY: -2, maxY: 2 },
      [[0, 1, 2, 0, false]],
      [],
    );
    const out = stretchRun(run, 800, 600);
    expect(out.bounds).toEqual({ minX: -400, maxX: 400, minY: -300, maxY: 300 });
  });

  it("maps the run's max corner to the stretched-bounds max corner", () => {
    const run = makeRun(
      { minX: -1, maxX: 1, minY: -2, maxY: 2 },
      [[0, 1, 2, 0, false]],
      [],
    );
    const out = stretchRun(run, 800, 600);
    // raw center = (0, 0); raw half-width = 1; sx = 800 / 2 = 400.
    // raw half-height = 2; sy = 600 / 4 = 150.
    // point (1, 2) → ((1-0)*400, (2-0)*150) = (400, 300).
    const [, x, y] = out.users[0]!;
    expect(x).toBeCloseTo(400, 6);
    expect(y).toBeCloseTo(300, 6);
  });

  it("stretches clusters via stretchEllipse (centered on the bounds midpoint)", () => {
    const run = makeRun(
      { minX: 0, maxX: 2, minY: 0, maxY: 4 },
      [],
      [{ id: 0, label: "A", cx: 1, cy: 2, rx: 0.5, ry: 1, angle: 0, size: 3, has_detail: false }],
    );
    const out = stretchRun(run, 200, 400);
    // raw center = (1, 2); sx = 200 / 2 = 100; sy = 400 / 4 = 100.
    // post-center, cluster center is (0, 0) → stretched center (0, 0).
    const c = out.clusters[0]!;
    expect(c.cx).toBeCloseTo(0, 6);
    expect(c.cy).toBeCloseTo(0, 6);
    expect(c.rx).toBeCloseTo(50, 6);
    expect(c.ry).toBeCloseTo(100, 6);
    expect(c.angle).toBeCloseTo(0, 9);
  });

  it("preserves the user tuple ids and cluster ids", () => {
    const run = makeRun(
      { minX: 0, maxX: 10, minY: 0, maxY: 10 },
      [[42, 5, 5, 7, false]],
      [{ id: 7, label: "lucky", cx: 5, cy: 5, rx: 1, ry: 1, angle: 0, size: 1, has_detail: false }],
    );
    const out = stretchRun(run, 100, 100);
    expect(out.users[0]![0]).toBe(42);
    expect(out.users[0]![3]).toBe(7);
    expect(out.clusters[0]!.id).toBe(7);
    expect(out.clusters[0]!.label).toBe("lucky");
    expect(out.clusters[0]!.size).toBe(1);
  });
});
