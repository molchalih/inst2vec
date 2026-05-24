import { easeOutCubic } from "./ease";
import { hashUnit } from "./hash";

export type IntroPhase = 0 | 1 | 2; // 0 fade, 1 flight, 2 settle

export type IntroDurations = {
  fadeMs: number;
  flightMs: number;
  settleMs: number;
};

/**
 * Maps elapsed-ms to the intro phase and that phase's local progress.
 * A zero-length phase reports progress 1 so a caller can advance past
 * it. `done` is true once the whole timeline has elapsed.
 */
export const introPhaseAndProgress = (
  elapsed: number,
  d: IntroDurations,
): { phase: IntroPhase; progress: number; done: boolean } => {
  const total = d.fadeMs + d.flightMs + d.settleMs;
  if (elapsed >= total) return { phase: 2, progress: 1, done: true };
  if (elapsed < d.fadeMs) {
    return { phase: 0, progress: d.fadeMs <= 0 ? 1 : elapsed / d.fadeMs, done: false };
  }
  if (elapsed < d.fadeMs + d.flightMs) {
    const e = elapsed - d.fadeMs;
    return { phase: 1, progress: d.flightMs <= 0 ? 1 : e / d.flightMs, done: false };
  }
  const e = elapsed - d.fadeMs - d.flightMs;
  return { phase: 2, progress: d.settleMs <= 0 ? 1 : e / d.settleMs, done: false };
};

/**
 * Per-dot eased flight progress in [0, 1]. Each dot waits a
 * deterministic random delay (hashUnit(id) * maxStaggerFrac) before
 * launching, then eases over the remaining window so all dots land at
 * flightProgress 1. maxStaggerFrac must be < 1.
 */
export const introStagger = (
  flightProgress: number,
  id: number,
  maxStaggerFrac: number,
): number => {
  const delay = hashUnit(id) * maxStaggerFrac;
  if (flightProgress <= delay) return 0;
  return easeOutCubic((flightProgress - delay) / (1 - delay));
};

/** Dot alpha: ramps 0 → baseAlpha during fade, full thereafter. */
export const introDotAlpha = (
  phase: IntroPhase,
  progress: number,
  baseAlpha: number,
): number => (phase === 0 ? baseAlpha * progress : baseAlpha);

/** Ellipse alpha scale: 0 until dots land, ramps 0 → 1 during settle. */
export const introEllipseAlpha = (phase: IntroPhase, progress: number): number =>
  phase < 2 ? 0 : progress;
