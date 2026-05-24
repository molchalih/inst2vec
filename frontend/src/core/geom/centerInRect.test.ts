import { describe, expect, it } from "vitest";
import { centerWorldPointInRect } from "./centerInRect";

describe("centerWorldPointInRect", () => {
  it("places the world point at the rect's center at the given scale", () => {
    const t = centerWorldPointInRect(
      { x: 5, y: 10 },
      { x: 100, y: 50, width: 400, height: 200 },
      2,
    );
    expect(t.scale).toBe(2);
    // Center of rect = (300, 150). At scale=2 the screen point for
    // world(5, 10) must equal (300, 150).
    expect(t.x + 5 * t.scale).toBeCloseTo(300);
    expect(t.y + 10 * t.scale).toBeCloseTo(150);
  });

  it("returns identity-shaped transform if scale is non-positive", () => {
    const t = centerWorldPointInRect(
      { x: 0, y: 0 },
      { x: 0, y: 0, width: 100, height: 100 },
      0,
    );
    expect(t).toEqual({ x: 0, y: 0, scale: 1 });
  });
});
