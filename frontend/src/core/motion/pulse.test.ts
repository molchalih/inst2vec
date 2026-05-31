import { describe, expect, it } from "vitest";
import { sinePulse } from "./pulse";

const PERIOD = 1600;
const MIN = 0.45;
const MAX = 1.0;

describe("sinePulse", () => {
  it("returns min at elapsedMs=0 (breath starts at the trough)", () => {
    expect(sinePulse(0, PERIOD, MIN, MAX)).toBeCloseTo(MIN, 6);
  });

  it("returns max at the half-period (crest)", () => {
    expect(sinePulse(PERIOD / 2, PERIOD, MIN, MAX)).toBeCloseTo(MAX, 6);
  });

  it("returns min again at the full period (seamless loop)", () => {
    expect(sinePulse(PERIOD, PERIOD, MIN, MAX)).toBeCloseTo(MIN, 6);
  });

  it("rises monotonically across the first half", () => {
    const a = sinePulse(PERIOD * 0.2, PERIOD, MIN, MAX);
    const b = sinePulse(PERIOD * 0.4, PERIOD, MIN, MAX);
    expect(b).toBeGreaterThan(a);
  });

  it("falls monotonically across the second half", () => {
    const a = sinePulse(PERIOD * 0.6, PERIOD, MIN, MAX);
    const b = sinePulse(PERIOD * 0.8, PERIOD, MIN, MAX);
    expect(b).toBeLessThan(a);
  });

  it("stays within [min, max] for inputs far past one period", () => {
    for (const frac of [0.1, 0.5, 0.9, 1.3, 2.7, 3.7, 10.25]) {
      const v = sinePulse(frac * PERIOD, PERIOD, MIN, MAX);
      expect(v).toBeGreaterThanOrEqual(MIN - 1e-9);
      expect(v).toBeLessThanOrEqual(MAX + 1e-9);
    }
  });

  it("repeats with the period (value at t equals value at t+period)", () => {
    const t = PERIOD * 0.37;
    expect(sinePulse(t, PERIOD, MIN, MAX)).toBeCloseTo(
      sinePulse(t + PERIOD, PERIOD, MIN, MAX),
      6,
    );
  });

  it("returns max for a degenerate non-positive period (guard)", () => {
    expect(sinePulse(123, 0, MIN, MAX)).toBe(MAX);
    expect(sinePulse(123, -50, MIN, MAX)).toBe(MAX);
  });
});
