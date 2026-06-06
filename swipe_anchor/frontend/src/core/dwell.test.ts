import { describe, expect, it } from "vitest";
import { accumulateDwell, sumDwell } from "./dwell";

describe("accumulateDwell", () => {
  it("adds elapsed time to a card without mutating the input", () => {
    const a = { 10: 400 };
    const b = accumulateDwell(a, 10, 250);
    expect(b).toEqual({ 10: 650 });
    expect(a).toEqual({ 10: 400 }); // immutable
  });

  it("starts a new card at the elapsed time", () => {
    expect(accumulateDwell({}, 20, 120)).toEqual({ 20: 120 });
  });

  it("ignores non-positive deltas", () => {
    expect(accumulateDwell({ 10: 100 }, 10, 0)).toEqual({ 10: 100 });
    expect(accumulateDwell({ 10: 100 }, 10, -5)).toEqual({ 10: 100 });
  });
});

describe("sumDwell", () => {
  it("totals all per-card dwell", () => {
    expect(sumDwell({ 10: 400, 20: 250 })).toBe(650);
    expect(sumDwell({})).toBe(0);
  });
});
