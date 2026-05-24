import { describe, expect, it } from "vitest";
import { add, sub, scale, len, dot } from "./vec2";

describe("vec2", () => {
  it("adds two vectors", () => {
    expect(add({ x: 1, y: 2 }, { x: 3, y: -1 })).toEqual({ x: 4, y: 1 });
  });
  it("subtracts two vectors", () => {
    expect(sub({ x: 5, y: 2 }, { x: 1, y: -1 })).toEqual({ x: 4, y: 3 });
  });
  it("scales a vector", () => {
    expect(scale({ x: 2, y: -3 }, 2)).toEqual({ x: 4, y: -6 });
  });
  it("computes length", () => {
    expect(len({ x: 3, y: 4 })).toBeCloseTo(5);
  });
  it("computes dot product", () => {
    expect(dot({ x: 1, y: 2 }, { x: 3, y: 4 })).toBe(11);
  });
});
