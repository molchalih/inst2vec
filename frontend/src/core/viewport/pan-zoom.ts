import type { Transform } from "../geom/transform";
import type { Vec2 } from "../geom/vec2";

export type ScaleBounds = { min: number; max: number };

const clamp = (v: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, v));

/**
 * Wheel zoom anchored on `cursor`. When `bounds` is supplied the scale
 * is clamped first and the translation is derived from the *effective*
 * factor (clamped_scale / current_scale). Without this, hitting a
 * scale limit would leave x/y computed from the unclamped factor,
 * causing the world point under the cursor to drift — the gesture
 * silently turns into a pan at the zoom ceiling/floor.
 */
export const applyWheel = (
  t: Transform,
  cursor: Vec2,
  factor: number,
  bounds?: ScaleBounds,
): Transform => {
  const targetScale = t.scale * factor;
  const nextScale = bounds ? clamp(targetScale, bounds.min, bounds.max) : targetScale;
  const effective = t.scale === 0 ? factor : nextScale / t.scale;
  return {
    scale: nextScale,
    x: cursor.x - (cursor.x - t.x) * effective,
    y: cursor.y - (cursor.y - t.y) * effective,
  };
};

export const applyDrag = (t: Transform, delta: Vec2): Transform => ({
  x: t.x + delta.x,
  y: t.y + delta.y,
  scale: t.scale,
});
