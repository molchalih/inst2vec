import { describe, expect, it } from "vitest";
import { isPointInEllipse } from "./ellipse";

describe("isPointInEllipse", () => {
  it("detects a point inside an axis-aligned ellipse", () => {
    expect(isPointInEllipse({ x: 1, y: 0 }, { cx: 0, cy: 0, rx: 2, ry: 1, angle: 0 })).toBe(true);
  });

  it("rejects a point outside", () => {
    expect(isPointInEllipse({ x: 3, y: 0 }, { cx: 0, cy: 0, rx: 2, ry: 1, angle: 0 })).toBe(false);
  });

  it("respects rotation", () => {
    const e = { cx: 0, cy: 0, rx: 2, ry: 1, angle: Math.PI / 2 };
    expect(isPointInEllipse({ x: 0, y: 1.5 }, e)).toBe(true);
    expect(isPointInEllipse({ x: 1.5, y: 0 }, e)).toBe(false);
  });
});
