import { describe, expect, it } from "vitest";
import { detectTired, median } from "./coaching";

describe("median", () => {
  it("handles odd and even lengths", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([1, 2, 3, 4])).toBe(2.5);
  });
});

describe("detectTired", () => {
  it("is false with no history", () => {
    expect(detectTired([])).toBe(false);
  });

  it("flags a single very long judgment regardless of sample count", () => {
    expect(detectTired([50_000])).toBe(true);
  });

  it("does not flag a slowdown until there are enough samples", () => {
    // latest is 3x the baseline but only 2 prior samples → not enough yet.
    expect(detectTired([3000, 3000, 30_000])).toBe(false);
  });

  it("flags a slowdown well past the baseline and the floor", () => {
    // base = median([3000,3500,4000]) = 3500; threshold = max(12000, 7700) = 12000.
    expect(detectTired([3000, 3500, 4000, 13_000])).toBe(true);
  });

  it("does not flag normal pace even if a bit slower", () => {
    // base 3500, latest 8000 — slower but under the 12s floor → fine.
    expect(detectTired([3000, 3500, 4000, 8000])).toBe(false);
  });

  it("does not flag a steady slow-but-consistent pace", () => {
    // Consistently ~13s: latest is not markedly slower than the baseline.
    expect(detectTired([13_000, 13_000, 13_000, 13_000])).toBe(false);
  });
});
