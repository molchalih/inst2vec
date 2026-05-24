import { describe, expect, it } from "vitest";
import { formatCompact } from "./compact";

describe("formatCompact", () => {
  it.each([
    [0, "0"],
    [42, "42"],
    [999, "999"],
    [1000, "1k"],
    [1100, "1.1k"],
    [18412, "18.4k"],
    [999900, "999.9k"],
    [1_000_000, "1M"],
    [1_500_000, "1.5M"],
    [9_999_999, "10M"],
    [100_000_000, "100M"],
  ])("%i → %s", (input, expected) => {
    expect(formatCompact(input)).toBe(expected);
  });

  it("handles negatives", () => {
    expect(formatCompact(-1500)).toBe("-1.5k");
  });
});
