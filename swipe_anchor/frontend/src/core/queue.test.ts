import { describe, expect, it } from "vitest";
import { shouldRefill } from "./queue";

describe("shouldRefill", () => {
  it("refills when at or below the low-water mark", () => {
    expect(shouldRefill(1, 2)).toBe(true);
    expect(shouldRefill(2, 2)).toBe(true);
    expect(shouldRefill(0, 2)).toBe(true);
  });

  it("does not refill when comfortably stocked", () => {
    expect(shouldRefill(3, 2)).toBe(false);
  });

  it("uses a default low-water mark of 2", () => {
    expect(shouldRefill(2)).toBe(true);
    expect(shouldRefill(3)).toBe(false);
  });
});
