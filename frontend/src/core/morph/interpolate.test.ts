import { describe, expect, it } from "vitest";
import { interpolateUsers, interpolateEllipses, lerpHex } from "./interpolate";
import type { JoinedCluster, JoinedUser } from "./join";

const palette = ["#d2d8d9", "#969293", "#e96f51", "#80c470"] as const;
const noise = "#576175";

describe("lerpHex", () => {
  it("returns the from color at t=0", () => {
    expect(lerpHex("#ff0000", "#0000ff", 0)).toBe("#ff0000");
  });
  it("returns the to color at t=1", () => {
    expect(lerpHex("#ff0000", "#0000ff", 1)).toBe("#0000ff");
  });
  it("blends channels linearly at t=0.5", () => {
    expect(lerpHex("#000000", "#ffffff", 0.5)).toBe("#808080");
  });
});

describe("interpolateUsers", () => {
  it("lerps position for creators present in both runs", () => {
    const joined: JoinedUser[] = [{
      id: 1,
      fromXY: [0, 0], toXY: [10, 20],
      fromCluster: 0, toCluster: 0,
      fromCentrality: 0, toCentrality: 0,
    }];
    const out = interpolateUsers(joined, () => 0.5, () => 0.5, palette, noise);
    expect(out[0]!.x).toBeCloseTo(5);
    expect(out[0]!.y).toBeCloseTo(10);
  });
  it("keeps from-only creator at fromXY with fading alpha", () => {
    const joined: JoinedUser[] = [{
      id: 1, fromXY: [3, 4], toXY: null,
      fromCluster: 0, toCluster: null,
      fromCentrality: 0, toCentrality: 0,
    }];
    const out = interpolateUsers(joined, () => 0.25, () => 0.25, palette, noise);
    expect(out[0]!.x).toBeCloseTo(3);
    expect(out[0]!.y).toBeCloseTo(4);
    expect(out[0]!.alpha).toBeCloseTo(0.75);
  });
  it("keeps to-only creator at toXY with rising alpha", () => {
    const joined: JoinedUser[] = [{
      id: 1, fromXY: null, toXY: [7, 8],
      fromCluster: null, toCluster: 0,
      fromCentrality: 0, toCentrality: 0,
    }];
    const out = interpolateUsers(joined, () => 0.25, () => 0.25, palette, noise);
    expect(out[0]!.x).toBeCloseTo(7);
    expect(out[0]!.y).toBeCloseTo(8);
    expect(out[0]!.alpha).toBeCloseTo(0.25);
  });
  it("lerps cluster color for creators present in both runs", () => {
    const joined: JoinedUser[] = [{
      id: 1, fromXY: [0, 0], toXY: [0, 0],
      fromCluster: 0, toCluster: 1,
      fromCentrality: 0, toCentrality: 0,
    }];
    const start = interpolateUsers(joined, () => 0, () => 0, palette, noise)[0]!.color;
    const mid = interpolateUsers(joined, () => 0.5, () => 0.5, palette, noise)[0]!.color;
    const end = interpolateUsers(joined, () => 1, () => 1, palette, noise)[0]!.color;
    expect(start).toMatch(/^#[0-9a-f]{6}$/i);
    expect(end).toMatch(/^#[0-9a-f]{6}$/i);
    expect(start).not.toBe(end);
    expect(mid).not.toBe(start);
    expect(mid).not.toBe(end);
  });
  it("evaluates progress per joined-user index", () => {
    const joined: JoinedUser[] = [
      { id: 0, fromXY: [0, 0], toXY: [10, 0], fromCluster: 0, toCluster: 0, fromCentrality: 0, toCentrality: 0 },
      { id: 1, fromXY: [0, 0], toXY: [10, 0], fromCluster: 0, toCluster: 0, fromCentrality: 0, toCentrality: 0 },
    ];
    const out = interpolateUsers(joined, (i) => i === 0 ? 0 : 1, (i) => i === 0 ? 0 : 1, palette, noise);
    expect(out[0]!.x).toBeCloseTo(0);
    expect(out[1]!.x).toBeCloseTo(10);
  });
  it("evaluates motion and color progress independently", () => {
    const joined: JoinedUser[] = [
      { id: 0, fromXY: [0, 0], toXY: [10, 0], fromCluster: 0, toCluster: 1, fromCentrality: 0, toCentrality: 0 },
    ];
    // motion=1 (fully arrived), color=0 (still from-color)
    const arrivedButFromColor = interpolateUsers(joined, () => 1, () => 0, palette, noise);
    expect(arrivedButFromColor[0]!.x).toBeCloseTo(10);
    // motion=0 (still at from), color=1 (already to-color)
    const stillButToColor = interpolateUsers(joined, () => 0, () => 1, palette, noise);
    expect(stillButToColor[0]!.x).toBeCloseTo(0);
    // Verify color paths differ:
    expect(arrivedButFromColor[0]!.color).not.toBe(stillButToColor[0]!.color);
  });
});

describe("lerpHex error handling", () => {
  it("throws on malformed hex input", () => {
    expect(() => lerpHex("not-a-color", "#000000", 0.5)).toThrow(/expected #rrggbb/);
    expect(() => lerpHex("#000000", "rgb(0,0,0)", 0.5)).toThrow(/expected #rrggbb/);
  });
});

describe("interpolateEllipses", () => {
  it("returns shape from the requested side", () => {
    const joined: JoinedCluster[] = [{
      id: 0,
      from: { cx: 0, cy: 0, rx: 1, ry: 1, angle: 0 },
      to: { cx: 10, cy: 10, rx: 2, ry: 2, angle: 0.5 },
    }];
    const fromSide = interpolateEllipses(joined, "from", palette, noise);
    expect(fromSide[0]!.cx).toBe(0);
    const toSide = interpolateEllipses(joined, "to", palette, noise);
    expect(toSide[0]!.cx).toBe(10);
  });
  it("omits clusters that don't exist on the requested side", () => {
    const joined: JoinedCluster[] = [
      { id: 0, from: { cx: 0, cy: 0, rx: 1, ry: 1, angle: 0 }, to: null },
      { id: 1, from: null, to: { cx: 5, cy: 5, rx: 1, ry: 1, angle: 0 } },
    ];
    expect(interpolateEllipses(joined, "from", palette, noise).map((e) => e.id)).toEqual([0]);
    expect(interpolateEllipses(joined, "to", palette, noise).map((e) => e.id)).toEqual([1]);
  });
});
