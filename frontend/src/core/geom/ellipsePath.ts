import type { Vec2 } from "./vec2";
import type { Ellipse } from "./ellipse";

/**
 * Returns `segments` points evenly distributed along the perimeter of
 * a rotated ellipse, starting at angle 0 in the ellipse's local frame
 * and proceeding counter-clockwise.
 *
 * Pure: no Pixi, no DOM. Consumers feed the result into `g.poly(...)`
 * or similar.
 */
export const ellipsePoints = (e: Ellipse, segments: number): Vec2[] => {
  const n = Math.max(3, segments | 0);
  const cos = Math.cos(e.angle);
  const sin = Math.sin(e.angle);
  const out: Vec2[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const t = (2 * Math.PI * i) / n;
    const lx = e.rx * Math.cos(t);
    const ly = e.ry * Math.sin(t);
    out[i] = {
      x: e.cx + lx * cos - ly * sin,
      y: e.cy + lx * sin + ly * cos,
    };
  }
  return out;
};
