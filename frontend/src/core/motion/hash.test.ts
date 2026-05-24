import { describe, expect, it } from "vitest";
import { hashUnit } from "./hash";

describe("hashUnit", () => {
  it("is deterministic for a given id", () => {
    expect(hashUnit(42)).toBe(hashUnit(42));
  });

  it("returns a value in [0, 1)", () => {
    for (const id of [0, 1, 2, 7, 99, 1000, 123456]) {
      const v = hashUnit(id);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("spreads nearby ids apart (not a ramp)", () => {
    expect(Math.abs(hashUnit(1) - hashUnit(2))).toBeGreaterThan(0.05);
  });
});
