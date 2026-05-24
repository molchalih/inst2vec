import type { Vec2 } from "../geom/vec2";
import type { Transform } from "../geom/transform";
import type { Rect } from "../geom/fitBoundsToRect";
import { centerWorldPointInRect } from "../geom/centerInRect";
import type { ClampLimits } from "./clamp";

const clamp = (v: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, v));

/**
 * Camera-focus target: place `center` at the centre of `rect` (the
 * panel-inset visible area), but cap the zoom to the same scale band the
 * clamp enforces ([min, max] × fitScale) and keep the point centred at
 * that capped scale.
 *
 * The cap matters: a small edge cluster's natural fit can want a zoom far
 * beyond the ceiling. Centring at the *uncapped* scale and letting the
 * per-frame clamp cap it later leaves the pan computed for the wrong
 * scale, so the cluster lands off-centre. Capping here keeps it centred.
 */
export const focusTransform = (
  center: Vec2,
  rect: Rect,
  desiredScale: number,
  fitScale: number,
  limits: ClampLimits,
): Transform => {
  const scale = clamp(
    desiredScale,
    limits.minScaleFactor * fitScale,
    limits.maxScaleFactor * fitScale,
  );
  return centerWorldPointInRect(center, rect, scale);
};
