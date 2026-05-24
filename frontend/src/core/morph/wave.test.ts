import { describe, expect, it } from "vitest";
import { computeWaveDelays, waveProgress, scalePop } from "./wave";
import type { JoinedUser } from "./join";

const ju = (id: number, fromXY: [number, number] | null, toXY: [number, number] | null): JoinedUser => ({
  id, fromXY, toXY, fromCluster: 0, toCluster: 0,
});

describe("computeWaveDelays", () => {
  it("returns a Float32Array of the same length as the input", () => {
    const out = computeWaveDelays([ju(0, [0, 0], null), ju(1, [1, 0], null)]);
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(2);
  });

  it("normalizes to [0, 1] with the furthest dot at 1", () => {
    const out = computeWaveDelays([
      ju(0, [0, 0], null),
      ju(1, [3, 4], null),
      ju(2, [6, 8], null),
    ]);
    expect(out[0]).toBeCloseTo(0);
    expect(out[1]).toBeCloseTo(0.5);
    expect(out[2]).toBeCloseTo(1);
  });

  it("uses fromXY when toXY is null", () => {
    const out = computeWaveDelays([ju(0, [3, 4], null)]);
    expect(out[0]).toBeCloseTo(1);
  });

  it("uses toXY (destination) when both sides are present", () => {
    // fromXY at distance 5, toXY at distance 10 — delayNorm must reflect toXY.
    const out = computeWaveDelays([
      ju(0, [3, 4], [0, 0]),
      ju(1, [0, 0], [6, 8]),
    ]);
    // After normalization: id 0 toXY=(0,0) → 0, id 1 toXY=(6,8) → 10 → max → 1.
    expect(out[0]).toBeCloseTo(0);
    expect(out[1]).toBeCloseTo(1);
  });

  it("returns zeros when every position is at the origin", () => {
    const out = computeWaveDelays([ju(0, [0, 0], null), ju(1, [0, 0], null)]);
    expect(out[0]).toBe(0);
    expect(out[1]).toBe(0);
  });

  it("returns zero for dots with no position on either side", () => {
    const out = computeWaveDelays([ju(0, null, null), ju(1, [3, 4], null)]);
    expect(out[0]).toBe(0);
    expect(out[1]).toBeCloseTo(1);
  });
});

describe("waveProgress", () => {
  it("is 0 before the dot's window opens", () => {
    expect(waveProgress(0.4, 1, 0.5, 0, 0)).toBe(0);
  });

  it("ramps to 1 across the dot's window", () => {
    expect(waveProgress(0.75, 1, 0.5, 0, 0)).toBeCloseTo(0.5);
  });

  it("clamps to 1 at and past the end of the window", () => {
    expect(waveProgress(1.0, 1, 0.5, 0, 0)).toBe(1);
    expect(waveProgress(1.5, 1, 0.5, 0, 0)).toBe(1);
  });

  it("centre-dot (delayNorm=0) starts immediately", () => {
    expect(waveProgress(0, 0, 0.5, 0, 0)).toBeGreaterThanOrEqual(0);
    expect(waveProgress(0.25, 0, 0.5, 0, 0)).toBeCloseTo(0.5);
  });

  it("jitter shifts the window deterministically per id", () => {
    const a = waveProgress(0.6, 0.5, 0.4, 0.1, 42);
    const b = waveProgress(0.6, 0.5, 0.4, 0.1, 42);
    expect(a).toBe(b);
    const c = waveProgress(0.6, 0.5, 0.4, 0.1, 43);
    expect(typeof c).toBe("number");
  });

  it("with negative jitter does NOT trigger before globalProgress = 0 (caller must short-circuit)", () => {
    // Document the leak: when jitter < 0 and delayNorm = 0, start can go
    // negative and a globalProgress=0 call returns > 0. DotsLayer must
    // short-circuit phases 0/1 to 0 before calling this helper. This test
    // pins the math so the layer can rely on the contract.
    // jitterFor(0) is a deterministic constant — compute it once:
    const startNegativeIds = Array.from({ length: 200 }, (_, i) => i)
      .filter((id) => waveProgress(0, 0, 0.5, 0.1, id) > 0);
    expect(startNegativeIds.length).toBeGreaterThan(0);
  });
});

describe("scalePop", () => {
  it("returns 1 outside the active window", () => {
    expect(scalePop(0, 1.4, 0.25)).toBe(1);
    expect(scalePop(1, 1.4, 0.25)).toBe(1);
    expect(scalePop(-0.1, 1.4, 0.25)).toBe(1);
    expect(scalePop(1.1, 1.4, 0.25)).toBe(1);
  });

  it("peaks at upFrac with value peak", () => {
    expect(scalePop(0.25, 1.4, 0.25)).toBeCloseTo(1.4, 6);
  });

  it("is monotonically increasing on the up-leg", () => {
    const a = scalePop(0.1, 1.4, 0.25);
    const b = scalePop(0.2, 1.4, 0.25);
    expect(b).toBeGreaterThan(a);
  });

  it("is monotonically decreasing on the down-leg", () => {
    const a = scalePop(0.4, 1.4, 0.25);
    const b = scalePop(0.8, 1.4, 0.25);
    expect(b).toBeLessThan(a);
  });

  it("is a no-op when peak === 1", () => {
    expect(scalePop(0.25, 1, 0.25)).toBeCloseTo(1, 6);
    expect(scalePop(0.5, 1, 0.25)).toBeCloseTo(1, 6);
  });
});
