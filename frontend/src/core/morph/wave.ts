import type { JoinedUser } from "./join";
import { hashUnit } from "../motion/hash";

// Deterministic wobble in [-0.5, 0.5), derived from the shared hash.
const jitterFor = (id: number): number => hashUnit(id) - 0.5;

/**
 * Normalized distance of each joined user's destination position from
 * the world origin, in [0, 1]. Computed once at transition seed; the
 * tick loop only multiplies by global progress. Using the destination
 * (toXY) makes the pulse fan outward from the new arrangement's centre
 * after the flight has landed.
 */
export const computeWaveDelays = (
  joined: ReadonlyArray<JoinedUser>,
): Float32Array => {
  const out = new Float32Array(joined.length);
  let max = 0;
  for (let i = 0; i < joined.length; i++) {
    const j = joined[i]!;
    const xy = j.toXY ?? j.fromXY;
    if (!xy) { out[i] = 0; continue; }
    const d = Math.hypot(xy[0], xy[1]);
    out[i] = d;
    if (d > max) max = d;
  }
  if (max > 0) {
    for (let i = 0; i < out.length; i++) out[i]! /= max;
  } else {
    for (let i = 0; i < out.length; i++) out[i] = 0;
  }
  return out;
};

/**
 * Localised per-dot progress through the wave.
 *
 *   globalProgress in [0, 1]   phase-2 progress
 *   delayNorm in [0, 1]        per-dot distance-from-origin
 *   spread in (0, 1]           width of the wavefront as a fraction of
 *                              total phase-2 duration (smaller = sharper)
 *   jitterAmp in [0, 0.5)      wobble fraction added to the start
 *   id                          dot id; provides deterministic jitter
 *
 * Each dot's window is [delayNorm*(1-spread) + jitter, +spread].
 */
export const waveProgress = (
  globalProgress: number,
  delayNorm: number,
  spread: number,
  jitterAmp: number,
  id: number,
): number => {
  const jitter = jitterFor(id) * jitterAmp;
  const start = delayNorm * (1 - spread) + jitter;
  if (globalProgress <= start) return 0;
  const local = (globalProgress - start) / spread;
  return Math.min(local, 1);
};

/**
 * Asymmetric scale pop: quick grow to `peak` at `upFrac` of the window,
 * slow ease back to 1 over the rest. Returns 1 outside [0, 1].
 */
export const scalePop = (
  localProgress: number,
  peak: number,
  upFrac: number,
): number => {
  if (localProgress <= 0 || localProgress >= 1) return 1;
  if (localProgress < upFrac) {
    const t = localProgress / upFrac;
    return 1 + (peak - 1) * (1 - (1 - t) * (1 - t));
  }
  const t = (localProgress - upFrac) / (1 - upFrac);
  return peak + (1 - peak) * t * t;
};
