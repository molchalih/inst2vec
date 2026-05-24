import type { Vec2 } from "./vec2";

export type Ellipse = { cx: number; cy: number; rx: number; ry: number; angle: number };

export const isPointInEllipse = (p: Vec2, e: Ellipse): boolean => {
  const dx = p.x - e.cx;
  const dy = p.y - e.cy;
  const cos = Math.cos(-e.angle);
  const sin = Math.sin(-e.angle);
  const lx = dx * cos - dy * sin;
  const ly = dx * sin + dy * cos;
  const rx = Math.max(e.rx, 1e-9);
  const ry = Math.max(e.ry, 1e-9);
  return (lx * lx) / (rx * rx) + (ly * ly) / (ry * ry) <= 1;
};
