import { describe, expect, it } from "vitest";
import { clampPanZoom } from "./clamp";

const bounds = { minX: -100, maxX: 100, minY: -50, maxY: 50 };
const size = { width: 400, height: 200 };
const fitScale = 2;
const limits = { minScaleFactor: 0.5, maxScaleFactor: 4, panMarginPx: 20 };

describe("clampPanZoom — scale", () => {
  it("clamps below the relative floor", () => {
    const t = clampPanZoom({ x: 200, y: 100, scale: 0.1 }, bounds, size, fitScale, limits);
    expect(t.scale).toBeCloseTo(0.5 * fitScale);
  });
  it("clamps above the relative ceiling", () => {
    const t = clampPanZoom({ x: 200, y: 100, scale: 1000 }, bounds, size, fitScale, limits);
    expect(t.scale).toBeCloseTo(4 * fitScale);
  });
  it("passes valid scales through", () => {
    const t = clampPanZoom({ x: 200, y: 100, scale: fitScale }, bounds, size, fitScale, limits);
    expect(t.scale).toBeCloseTo(fitScale);
  });
});

describe("clampPanZoom — pan", () => {
  it("keeps the atlas at least panMarginPx visible when pushed right", () => {
    const t = clampPanZoom({ x: 10_000, y: 100, scale: fitScale }, bounds, size, fitScale, limits);
    expect(bounds.minX * t.scale + t.x).toBeLessThanOrEqual(size.width - limits.panMarginPx + 1e-6);
  });
  it("keeps the atlas at least panMarginPx visible when pushed left", () => {
    const t = clampPanZoom({ x: -10_000, y: 100, scale: fitScale }, bounds, size, fitScale, limits);
    expect(bounds.maxX * t.scale + t.x).toBeGreaterThanOrEqual(limits.panMarginPx - 1e-6);
  });
  it("clamps to the visible-side boundary (atlas right edge at panMarginPx)", () => {
    const xAtLimit = limits.panMarginPx - bounds.maxX * fitScale;
    const t = clampPanZoom({ x: xAtLimit - 5, y: 0, scale: fitScale }, bounds, size, fitScale, limits);
    expect(t.x).toBeCloseTo(xAtLimit);
  });
});
