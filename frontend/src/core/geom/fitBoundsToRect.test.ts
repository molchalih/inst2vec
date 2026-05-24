import { describe, expect, it } from "vitest";
import { fitBoundsToRect } from "./fitBoundsToRect";

describe("fitBoundsToRect", () => {
  it("centers bounds inside the rect (no offset) at the given padding", () => {
    const bounds = { minX: -10, maxX: 10, minY: -5, maxY: 5 };
    const rect = { x: 0, y: 0, width: 800, height: 400 };
    const t = fitBoundsToRect(bounds, rect, 0); // padding=0 → fills fully
    // Scale chosen to fit the smaller axis (height); 400 / 10 = 40.
    expect(t.scale).toBeCloseTo(40);
    // Center of bounds (cx=0, cy=0) maps to center of rect (400, 200).
    expect(t.x).toBeCloseTo(400);
    expect(t.y).toBeCloseTo(200);
  });

  it("respects padding by shrinking the usable rect", () => {
    const bounds = { minX: -10, maxX: 10, minY: -10, maxY: 10 };
    const rect = { x: 0, y: 0, width: 100, height: 100 };
    const t = fitBoundsToRect(bounds, rect, 0.1); // 10% inset per side → 80x80
    expect(t.scale).toBeCloseTo(4);
  });

  it("offsets center by rect.x and rect.y", () => {
    const bounds = { minX: -10, maxX: 10, minY: -5, maxY: 5 };
    const rect = { x: 200, y: 50, width: 800, height: 400 };
    const t = fitBoundsToRect(bounds, rect, 0);
    // Center of rect is (200 + 400, 50 + 200) = (600, 250).
    expect(t.x).toBeCloseTo(600);
    expect(t.y).toBeCloseTo(250);
  });

  it("returns identity when the rect is degenerate", () => {
    const bounds = { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    const t = fitBoundsToRect(bounds, { x: 0, y: 0, width: 0, height: 100 }, 0);
    expect(t).toEqual({ x: 0, y: 0, scale: 1 });
  });
});
