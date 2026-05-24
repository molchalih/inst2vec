import type { Bounds } from "../geom/fit";
import type { Transform } from "../geom/transform";
import type { Rect } from "../geom/fitBoundsToRect";

export type ViewportSize = { width: number; height: number };

export type ClampLimits = {
  minScaleFactor: number;
  maxScaleFactor: number;
  panMarginPx: number;
};

const clamp = (v: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, v));

/**
 * Clamp the pan so the world `bounds` stay at least `m` px inside the
 * `region` on every side. When the bounds are smaller than the region
 * the allowed range collapses to a single value (centred).
 */
const clampPanToRegion = (
  t: Transform,
  bounds: Bounds,
  region: Rect,
  scale: number,
  m: number,
): { x: number; y: number } => {
  const xLow = region.x + m - bounds.maxX * scale;
  const xHigh = region.x + region.width - m - bounds.minX * scale;
  const x = xLow <= xHigh ? clamp(t.x, xLow, xHigh) : (xLow + xHigh) / 2;

  const yLow = region.y + m - bounds.maxY * scale;
  const yHigh = region.y + region.height - m - bounds.minY * scale;
  const y = yLow <= yHigh ? clamp(t.y, yLow, yHigh) : (yLow + yHigh) / 2;

  return { x, y };
};

/**
 * Clamp a candidate transform against the run's world bounds.
 *
 * Scale is clamped to [minScaleFactor, maxScaleFactor] × fitScale.
 * Pan is clamped so that the atlas stays at least `panMarginPx`
 * inside the viewport on every side — the visible-side edge of the
 * bounds can never cross to the far side of the viewport. When the
 * bounds are smaller than the visible window the allowed range
 * collapses to a single value (centred).
 */
export const clampPanZoom = (
  t: Transform,
  bounds: Bounds,
  size: ViewportSize,
  fitScale: number,
  limits: ClampLimits,
): Transform => {
  if (fitScale <= 0 || size.width <= 0 || size.height <= 0) return t;
  const scale = clamp(t.scale, limits.minScaleFactor * fitScale, limits.maxScaleFactor * fitScale);
  const region: Rect = { x: 0, y: 0, width: size.width, height: size.height };
  const { x, y } = clampPanToRegion(t, bounds, region, scale, limits.panMarginPx);
  return { x, y, scale };
};
