import type { Vec2 } from "./vec2";
import type { Transform } from "./transform";
import type { Rect } from "./fitBoundsToRect";

const IDENTITY: Transform = { x: 0, y: 0, scale: 1 };

/**
 * Build a Transform that lands the world point `p` at the rect's center
 * at the given uniform scale. Used by useCameraFocus for the "creator"
 * selection kind.
 */
export const centerWorldPointInRect = (
  p: Vec2,
  rect: Rect,
  scale: number,
): Transform => {
  if (scale <= 0) return IDENTITY;
  return {
    x: rect.x + rect.width / 2 - p.x * scale,
    y: rect.y + rect.height / 2 - p.y * scale,
    scale,
  };
};
