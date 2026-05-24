import type { Transform } from "./transform";
import type { Bounds } from "./fit";

export type Rect = { x: number; y: number; width: number; height: number };

const IDENTITY: Transform = { x: 0, y: 0, scale: 1 };

/**
 * Like fitBounds, but centers within an offset sub-rectangle of the
 * canvas. `padding` is a per-side fraction of the rect (0.1 = 10% inset
 * on each side). Centers bounds at (rect.x + rect.width / 2,
 * rect.y + rect.height / 2).
 */
export const fitBoundsToRect = (
  b: Bounds,
  rect: Rect,
  padding: number,
): Transform => {
  if (rect.width <= 0 || rect.height <= 0) return IDENTITY;
  const w = Math.max(b.maxX - b.minX, 1e-6);
  const h = Math.max(b.maxY - b.minY, 1e-6);
  const usableW = rect.width * (1 - 2 * padding);
  const usableH = rect.height * (1 - 2 * padding);
  const scale = Math.min(usableW / w, usableH / h);
  const cx = (b.minX + b.maxX) / 2;
  const cy = (b.minY + b.maxY) / 2;
  return {
    x: rect.x + rect.width / 2 - cx * scale,
    y: rect.y + rect.height / 2 - cy * scale,
    scale,
  };
};
