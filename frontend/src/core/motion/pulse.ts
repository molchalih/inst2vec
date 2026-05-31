/**
 * Continuous sine "breathing" — maps elapsed time onto [min, max] on a
 * seamless loop. Starts at `min` at elapsedMs=0, crests at `max` at the
 * half-period, returns to `min` at the full period. Pure and framework-free
 * (the rAF wiring lives in interaction/useSinePulse). A non-positive period
 * is degenerate and pins to `max` to avoid a divide-by-zero.
 */
export function sinePulse(
  elapsedMs: number,
  periodMs: number,
  min: number,
  max: number,
): number {
  if (periodMs <= 0) return max;
  const phase = (elapsedMs / periodMs) * Math.PI * 2;
  const unit = (1 - Math.cos(phase)) / 2;
  return min + (max - min) * unit;
}
