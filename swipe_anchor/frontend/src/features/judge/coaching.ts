/**
 * Pure coaching signals derived from recent reaction times. Kept out of the
 * component so the thresholds are unit-testable.
 */

// A single judgment this long means they're labouring over it.
export const TIRED_ABS_MS = 45_000;
// Need a few samples before calling fatigue from a slowdown trend.
export const TIRED_MIN_SAMPLES = 4;
// Below this a "slow" answer is still well within normal — don't nag.
export const TIRED_FLOOR_MS = 12_000;
// How much slower than their own baseline counts as flagging.
export const TIRED_RATIO = 2.2;

export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

/**
 * True when the latest answer shows fatigue: either a single very long one, or
 * one markedly slower than the person's own recent baseline. `history` ends with
 * the most recent reaction time (ms).
 */
export function detectTired(history: number[]): boolean {
  if (history.length === 0) return false;
  const latest = history[history.length - 1]!;
  if (latest >= TIRED_ABS_MS) return true;
  if (history.length < TIRED_MIN_SAMPLES) return false;
  const base = median(history.slice(0, -1));
  return latest > Math.max(TIRED_FLOOR_MS, base * TIRED_RATIO);
}
