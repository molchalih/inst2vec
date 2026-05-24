import type { Vec2 } from "./vec2";

export type Transform = { x: number; y: number; scale: number };

export const worldToScreen = (w: Vec2, t: Transform): Vec2 => ({
  x: w.x * t.scale + t.x,
  y: w.y * t.scale + t.y,
});

export const screenToWorld = (s: Vec2, t: Transform): Vec2 => ({
  x: (s.x - t.x) / t.scale,
  y: (s.y - t.y) / t.scale,
});
