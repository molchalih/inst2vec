import { describe, expect, it } from "vitest";
import type { AtlasRun, ClusterShape } from "@/data";
import { tokens } from "@/ui/tokens";
import { runToDotsFrame, runToEllipsesFrame, runToIntroDotsFrame } from "./frame";

const cluster = (over: Partial<ClusterShape> = {}): ClusterShape => ({
  id: 0, label: "c", cx: 0, cy: 0, rx: 1, ry: 1, angle: 0, size: 1, has_detail: false, ...over,
});

const makeRun = (
  users: AtlasRun["users"],
  clusters: ClusterShape[],
): AtlasRun => ({
  meta: { id: "video-1", case: "video", label: "v", size: users.length, details_available: false },
  bounds: { minX: -1, maxX: 1, minY: -1, maxY: 1 },
  users,
  clusters,
});

describe("runToDotsFrame", () => {
  it("returns an empty frame for a null run", () => {
    const f = runToDotsFrame(null);
    expect(f.users).toHaveLength(0);
    expect(f.alphaScale).toBe(1);
    expect(f.radiusScale).toBe(1);
  });

  it("maps each user row to a drawable, with signal alpha tokens.dot.alpha and a non-empty color", () => {
    const run = makeRun(
      [[0, 1, 2, 0, false], [1, 3, 4, 1, false]],
      [],
    );
    const f = runToDotsFrame(run);
    expect(f.users).toHaveLength(2);
    expect(f.users[0]).toMatchObject({ id: 0, x: 1, y: 2, alpha: tokens.dot.alpha });
    expect(f.users[1]).toMatchObject({ id: 1, x: 3, y: 4, alpha: tokens.dot.alpha });
    expect(f.users[0]!.color).toMatch(/^#[0-9a-f]{6}$/i);
    expect(f.users[0]!.color).not.toBe(f.users[1]!.color);
  });

  it("resolves noise users to the noise color + dimmed alpha", () => {
    const run = makeRun([[0, 0, 0, 0, false], [1, 0, 0, -1, false]], []);
    const f = runToDotsFrame(run);
    expect(f.users[0]!.alpha).toBe(tokens.dot.alpha);
    expect(f.users[1]!.alpha).toBeLessThan(f.users[0]!.alpha);
    expect(f.users[1]!.color).not.toBe(f.users[0]!.color);
  });
});

describe("runToEllipsesFrame", () => {
  it("returns an empty frame for a null run", () => {
    const f = runToEllipsesFrame(null);
    expect(f.ellipses).toHaveLength(0);
    expect(f.alphaScale).toBe(1);
    expect(f.strokeWidthScale).toBe(1);
  });

  it("includes only non-noise clusters", () => {
    const run = makeRun([], [cluster({ id: -1 }), cluster({ id: 0 }), cluster({ id: 1 })]);
    const f = runToEllipsesFrame(run);
    expect(f.ellipses).toHaveLength(2);
    expect(f.ellipses.map((e) => e.id)).toEqual([0, 1]);
  });

  it("resolves cluster color upfront", () => {
    const run = makeRun([], [cluster({ id: 0 }), cluster({ id: 1 })]);
    const f = runToEllipsesFrame(run);
    expect(typeof f.ellipses[0]!.color).toBe("string");
    expect(f.ellipses[0]!.color).not.toBe(f.ellipses[1]!.color);
  });
});

describe("runToIntroDotsFrame", () => {
  const center = { x: 100, y: 100 };

  it("returns an empty frame for a null run", () => {
    const f = runToIntroDotsFrame(null, center, 0, 0);
    expect(f.users).toHaveLength(0);
  });

  it("stacks every dot at centerWorld during the fade phase", () => {
    const run = makeRun([[0, 1, 2, 0, false], [1, 3, 4, 1, false]], []);
    const f = runToIntroDotsFrame(run, center, 0, 0.5);
    for (const u of f.users) {
      expect(u.x).toBeCloseTo(center.x, 5);
      expect(u.y).toBeCloseTo(center.y, 5);
    }
  });

  it("places every dot at its true position once flight completes (phase 2)", () => {
    const run = makeRun([[0, 1, 2, 0, false], [1, 3, 4, 1, false]], []);
    const f = runToIntroDotsFrame(run, center, 2, 1);
    expect(f.users[0]).toMatchObject({ x: 1, y: 2 });
    expect(f.users[1]).toMatchObject({ x: 3, y: 4 });
  });

  it("fades dot alpha in during the fade phase", () => {
    const run = makeRun([[0, 1, 2, 0, false]], []);
    const mid = runToIntroDotsFrame(run, center, 0, 0.5);
    expect(mid.users[0]!.alpha).toBeCloseTo(tokens.dot.alpha * 0.5, 5);
  });
});
