import { describe, expect, it } from "vitest";
import { flightProgress, interpolatedUserPos } from "./track-pos";
import type { JoinedUser } from "./join";

const ju = (over: Partial<JoinedUser>): JoinedUser => ({
  id: 1,
  fromXY: null,
  toXY: null,
  fromCluster: null,
  toCluster: null,
  fromCentrality: 0,
  toCentrality: 0,
  ...over,
});

describe("flightProgress", () => {
  const flightFrac = 0.5;

  it("is 0 before the flight window (phases 0 and 1)", () => {
    expect(flightProgress(0, 0.4, flightFrac)).toBe(0);
    expect(flightProgress(1, 0.9, flightFrac)).toBe(0);
  });

  it("is 1 after the flight window (phase 3)", () => {
    expect(flightProgress(3, 0.1, flightFrac)).toBe(1);
  });

  it("eases progress/flightFrac within phase 2, clamped to 1", () => {
    // at progress == flightFrac the raw fraction is 1 → eased 1
    expect(flightProgress(2, flightFrac, flightFrac)).toBeCloseTo(1, 6);
    // past flightFrac stays clamped at 1
    expect(flightProgress(2, 0.9, flightFrac)).toBeCloseTo(1, 6);
  });

  it("is monotonic and eased (easeOutCubic) inside phase 2", () => {
    const a = flightProgress(2, 0.1, flightFrac);
    const b = flightProgress(2, 0.2, flightFrac);
    expect(a).toBeGreaterThan(0);
    expect(b).toBeGreaterThan(a);
    // easeOutCubic(0.2) for raw 0.4 vs linear 0.4: eased is ahead of linear
    expect(flightProgress(2, 0.2, flightFrac)).toBeGreaterThan(0.4);
  });
});

describe("interpolatedUserPos", () => {
  const joined: JoinedUser[] = [
    ju({ id: 1, fromXY: [0, 0], toXY: [10, 20] }),
    ju({ id: 2, fromXY: [5, 5], toXY: null }),
    ju({ id: 3, fromXY: null, toXY: [7, 8] }),
  ];

  it("returns null for an absent creator id", () => {
    expect(interpolatedUserPos(joined, 99, 0.5)).toBeNull();
  });

  it("lerps both-sides position at the given motion progress", () => {
    expect(interpolatedUserPos(joined, 1, 0)).toEqual([0, 0]);
    expect(interpolatedUserPos(joined, 1, 1)).toEqual([10, 20]);
    expect(interpolatedUserPos(joined, 1, 0.5)).toEqual([5, 10]);
  });

  it("returns the from-side position when present from-only", () => {
    expect(interpolatedUserPos(joined, 2, 0.7)).toEqual([5, 5]);
  });

  it("returns the to-side position when present to-only", () => {
    expect(interpolatedUserPos(joined, 3, 0.3)).toEqual([7, 8]);
  });

  it("returns null for a row absent on both sides", () => {
    const both: JoinedUser[] = [ju({ id: 4, fromXY: null, toXY: null })];
    expect(interpolatedUserPos(both, 4, 0.5)).toBeNull();
  });
});
