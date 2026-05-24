import type { Transform } from "./transform";

export type Bounds = { minX: number; maxX: number; minY: number; maxY: number };
export type Viewport = { width: number; height: number };

const IDENTITY: Transform = { x: 0, y: 0, scale: 1 };

export const fitBounds = (b: Bounds, v: Viewport, padding = 0.8): Transform => {
  if (v.width <= 0 || v.height <= 0) return IDENTITY;
  const w = Math.max(b.maxX - b.minX, 1e-6);
  const h = Math.max(b.maxY - b.minY, 1e-6);
  const scale = padding * Math.min(v.width / w, v.height / h);
  const cx = (b.minX + b.maxX) / 2;
  const cy = (b.minY + b.maxY) / 2;
  return {
    x: v.width / 2 - cx * scale,
    y: v.height / 2 - cy * scale,
    scale,
  };
};
