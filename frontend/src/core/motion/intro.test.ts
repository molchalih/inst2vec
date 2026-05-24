import { describe, expect, it } from "vitest";
import {
  introPhaseAndProgress, introStagger, introDotAlpha, introEllipseAlpha,
} from "./intro";

const D = { fadeMs: 1000, flightMs: 2000, settleMs: 500 };

describe("introPhaseAndProgress", () => {
  it("is phase 0 (fade) during the fade window", () => {
    expect(introPhaseAndProgress(0, D)).toEqual({ phase: 0, progress: 0, done: false });
    expect(introPhaseAndProgress(500, D)).toMatchObject({ phase: 0, progress: 0.5 });
  });

  it("is phase 1 (flight) during the flight window", () => {
    expect(introPhaseAndProgress(1000, D)).toMatchObject({ phase: 1, progress: 0 });
    expect(introPhaseAndProgress(2000, D)).toMatchObject({ phase: 1, progress: 0.5 });
  });

  it("is phase 2 (settle) during the settle window", () => {
    expect(introPhaseAndProgress(3000, D)).toMatchObject({ phase: 2, progress: 0 });
    expect(introPhaseAndProgress(3250, D)).toMatchObject({ phase: 2, progress: 0.5 });
  });

  it("reports done at/after the total duration", () => {
    expect(introPhaseAndProgress(3500, D)).toEqual({ phase: 2, progress: 1, done: true });
    expect(introPhaseAndProgress(9999, D)).toEqual({ phase: 2, progress: 1, done: true });
  });
});

describe("introStagger", () => {
  it("returns 0 before a dot's delay has elapsed", () => {
    // hashUnit(id) * 0.4 is the delay; at flightProgress 0 nothing has launched.
    expect(introStagger(0, 1, 0.4)).toBe(0);
  });

  it("reaches 1 at flightProgress 1 for every dot", () => {
    for (const id of [0, 1, 5, 42, 999]) {
      expect(introStagger(1, id, 0.4)).toBeCloseTo(1, 5);
    }
  });

  it("is monotonic non-decreasing in flightProgress", () => {
    let prev = -1;
    for (let p = 0; p <= 1.0001; p += 0.1) {
      const v = introStagger(Math.min(p, 1), 7, 0.4);
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
    }
  });
});

describe("introDotAlpha", () => {
  it("ramps 0 → baseAlpha during fade (phase 0)", () => {
    expect(introDotAlpha(0, 0, 0.7)).toBeCloseTo(0, 5);
    expect(introDotAlpha(0, 0.5, 0.7)).toBeCloseTo(0.35, 5);
    expect(introDotAlpha(0, 1, 0.7)).toBeCloseTo(0.7, 5);
  });

  it("is full baseAlpha during flight and settle", () => {
    expect(introDotAlpha(1, 0, 0.7)).toBe(0.7);
    expect(introDotAlpha(2, 0.5, 0.7)).toBe(0.7);
  });
});

describe("introEllipseAlpha", () => {
  it("is 0 through fade and flight", () => {
    expect(introEllipseAlpha(0, 0.9)).toBe(0);
    expect(introEllipseAlpha(1, 0.9)).toBe(0);
  });

  it("ramps 0 → 1 during settle (phase 2)", () => {
    expect(introEllipseAlpha(2, 0)).toBe(0);
    expect(introEllipseAlpha(2, 1)).toBe(1);
  });
});
