import type { Ellipse } from "./ellipse";

/**
 * Anisotropic scale of a rotated ellipse. Center scales per axis;
 * semi-axes scale per axis; rotation rotates toward the more-stretched
 * direction via atan2(sy*sin, sx*cos). This is the Atlas-style
 * approximation — exact when angle is 0 or when sx == sy, and visually
 * adequate elsewhere for cluster shapes that already enclose their
 * dots loosely.
 *
 * Precondition: sx > 0 and sy > 0. Reflections (negative scales) are
 * not supported — semi-axes would become negative and downstream
 * consumers (`isPointInEllipse`, `ellipsePoints`) assume positive
 * extents.
 */
export const stretchEllipse = (e: Ellipse, sx: number, sy: number): Ellipse => ({
  cx: e.cx * sx,
  cy: e.cy * sy,
  rx: e.rx * sx,
  ry: e.ry * sy,
  angle: Math.atan2(sy * Math.sin(e.angle), sx * Math.cos(e.angle)),
});
