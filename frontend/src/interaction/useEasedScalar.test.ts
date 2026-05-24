import { describe, expect, it } from "vitest";
import { useEasedScalar } from "./useEasedScalar";

describe("useEasedScalar", () => {
  it("is a function with three parameters", () => {
    expect(typeof useEasedScalar).toBe("function");
    expect(useEasedScalar.length).toBe(3);
  });
});
