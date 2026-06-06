import { describe, expect, it } from "vitest";
import { deriveTriplets } from "./triplets";

describe("deriveTriplets", () => {
  it("turns one odd-one-out answer into two triplets", () => {
    expect(deriveTriplets([10, 20, 30], 30)).toEqual([
      { anchor: 10, positive: 20, negative: 30 },
      { anchor: 20, positive: 10, negative: 30 },
    ]);
  });

  it("works when the odd creator is the first", () => {
    expect(deriveTriplets([10, 20, 30], 10)).toEqual([
      { anchor: 20, positive: 30, negative: 10 },
      { anchor: 30, positive: 20, negative: 10 },
    ]);
  });

  it("rejects an odd id outside the triple", () => {
    expect(() => deriveTriplets([10, 20, 30], 99)).toThrow(/odd/i);
  });

  it("rejects a triple without three distinct creators", () => {
    expect(() => deriveTriplets([10, 10, 30], 30)).toThrow(/distinct/i);
  });
});
