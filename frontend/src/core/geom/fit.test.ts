import { describe, expect, it } from "vitest";
import { fitBounds } from "./fit";

describe("fitBounds", () => {
  it("centers and scales bounds to fit viewport with padding", () => {
    const bounds = { minX: -10, maxX: 10, minY: -5, maxY: 5 };
    const viewport = { width: 800, height: 400 };
    const t = fitBounds(bounds, viewport, 0.8);
    expect(t.scale).toBeCloseTo(32);
    expect(t.x).toBeCloseTo(400);
    expect(t.y).toBeCloseTo(200);
  });

  it("never returns scale 0 when bounds collapse", () => {
    const t = fitBounds({ minX: 0, maxX: 0, minY: 0, maxY: 0 }, { width: 100, height: 100 }, 0.8);
    expect(t.scale).toBeGreaterThan(0);
  });

  it("defaults padding to 0.8 when omitted", () => {
    const bounds = { minX: -10, maxX: 10, minY: -5, maxY: 5 };
    const viewport = { width: 800, height: 400 };
    const explicit = fitBounds(bounds, viewport, 0.8);
    const implicit = fitBounds(bounds, viewport);
    expect(implicit.scale).toBeCloseTo(explicit.scale);
    expect(implicit.x).toBeCloseTo(explicit.x);
    expect(implicit.y).toBeCloseTo(explicit.y);
  });

  it("returns identity transform when viewport has zero dimensions", () => {
    const t = fitBounds({ minX: -1, maxX: 1, minY: -1, maxY: 1 }, { width: 0, height: 0 });
    expect(t).toEqual({ x: 0, y: 0, scale: 1 });
  });

  it("returns identity transform when only width is zero", () => {
    const t = fitBounds({ minX: -1, maxX: 1, minY: -1, maxY: 1 }, { width: 0, height: 100 });
    expect(t).toEqual({ x: 0, y: 0, scale: 1 });
  });

  it("returns identity transform when only height is zero", () => {
    const t = fitBounds({ minX: -1, maxX: 1, minY: -1, maxY: 1 }, { width: 100, height: 0 });
    expect(t).toEqual({ x: 0, y: 0, scale: 1 });
  });
});
