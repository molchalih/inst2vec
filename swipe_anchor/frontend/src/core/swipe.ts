/**
 * Swipe geometry for the reel pager (plan §8.1, §8.2 core). Pure math; the React
 * gesture glue lives in interaction/.
 *
 * Horizontal swipes browse the three reels (left = forward, right = back);
 * a vertical swipe up is the commit gesture. The dominant axis wins so a mostly
 * horizontal drag never reads as a vertical one.
 */
export type SwipeDir = "left" | "right" | "up" | "down";

export interface SwipeVec {
  dx: number;
  dy: number;
  vx: number;
  vy: number;
  width: number;
  height: number;
}

export interface SwipeOpts {
  /** fraction of the axis length that counts as a committed drag (default 0.16) */
  distanceFrac?: number;
  /** velocity (px/ms) that counts as a flick (default 0.35) */
  flickVelocity?: number;
}

const clamp = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, n));

export function swipeGesture(v: SwipeVec, opts: SwipeOpts = {}): SwipeDir | null {
  const distanceFrac = opts.distanceFrac ?? 0.16;
  const flickVelocity = opts.flickVelocity ?? 0.35;
  const horizontal = Math.abs(v.dx) >= Math.abs(v.dy);
  const delta = horizontal ? v.dx : v.dy;
  const vel = horizontal ? v.vx : v.vy;
  const size = horizontal ? v.width : v.height;

  const byDistance = Math.abs(delta) > size * distanceFrac;
  const byVelocity = Math.abs(vel) > flickVelocity;
  if (!byDistance && !byVelocity) return null;

  const ref = byDistance ? delta : vel;
  if (horizontal) return ref < 0 ? "left" : "right";
  return ref < 0 ? "up" : "down";
}

/**
 * Map a gesture's speed to a confidence in [0,1]. A faster, more decisive motion
 * reads as a more confident judgment (plan §8.3).
 */
export function gestureConfidence(v: number, full = 1.2): number {
  return clamp(Math.abs(v) / full, 0, 1);
}
