import { useLayoutEffect, useRef } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  transitionAtom, transitionDriverAtom, viewportAtom, viewportSizeAtom,
  selectionAtom,
  type PhaseDurations, type TransitionPhase,
} from "@/state";
import { easeOutCubic, type Transform } from "@/core";
import { tokens } from "@/ui/tokens";

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
const lerpTransform = (a: Transform, b: Transform, t: number): Transform => ({
  x: lerp(a.x, b.x, t),
  y: lerp(a.y, b.y, t),
  scale: lerp(a.scale, b.scale, t),
});

// Numeric tolerance for "viewport already at target" — tighter than a
// screen pixel at any sensible scale, looser than rAF float wobble.
const TRANSFORM_EPSILON = 1e-3;

const transformsMatch = (a: Transform, b: Transform): boolean =>
  Math.abs(a.x - b.x) < TRANSFORM_EPSILON
  && Math.abs(a.y - b.y) < TRANSFORM_EPSILON
  && Math.abs(a.scale - b.scale) < TRANSFORM_EPSILON;

// Build per-switch durations from tokens, collapsing the camera phase
// to zero when the user hasn't panned/zoomed. Computed at effect
// setup, not per frame, since both inputs are constant for the run.
const buildDurations = (
  startTransform: Transform,
  targetTransform: Transform,
): PhaseDurations => {
  const v = tokens.motion.versionSwitch;
  return {
    phase0: transformsMatch(startTransform, targetTransform) ? 0 : v.phase1,
    phase1: v.phase2,
    phase2: v.phase3.flightMs + v.phase3.pulseMs,
    phase3: v.phase4,
  };
};

export const phaseAndProgress = (
  elapsed: number,
  d: PhaseDurations,
): { phase: TransitionPhase; progress: number; done: boolean } => {
  const total = d.phase0 + d.phase1 + d.phase2 + d.phase3;
  if (elapsed >= total) return { phase: 3, progress: 1, done: true };
  const thresholds = [d.phase0, d.phase0 + d.phase1, d.phase0 + d.phase1 + d.phase2];
  let phase: TransitionPhase = 0;
  let start = 0;
  if (elapsed >= thresholds[2]!) { phase = 3; start = thresholds[2]!; }
  else if (elapsed >= thresholds[1]!) { phase = 2; start = thresholds[1]!; }
  else if (elapsed >= thresholds[0]!) { phase = 1; start = thresholds[0]!; }
  const durs: Record<TransitionPhase, number> = {
    0: d.phase0, 1: d.phase1, 2: d.phase2, 3: d.phase3,
  };
  // A phase whose duration was collapsed to 0 (e.g. camera skip) is
  // treated as instantly-complete so phaseAndProgress can advance past it.
  const dur = durs[phase];
  const progress = dur <= 0 ? 1 : Math.min((elapsed - start) / dur, 1);
  return { phase, progress, done: false };
};

/**
 * Drives the four-phase version-switch animation: camera reset →
 * cluster fade-out → dot morph → cluster fade-in. Pure rAF consumer of
 * transitionDriverAtom — the driver itself is seeded atomically inside
 * ensureRunAtom, in the same synchronous batch as the activeRunId flip,
 * so the destination atlas never paints with transitionAtom still null.
 *
 * Per-switch durations are computed here (the interaction layer is
 * allowed to read ui/tokens; state/ is not) and the camera phase is
 * zeroed when the user hasn't panned/zoomed.
 *
 * useLayoutEffect (not useEffect) so the rAF schedule commits before
 * paint. Cleanup only cancels the frame; it does NOT null transition /
 * driver — strict-mode double-mount would otherwise erase a freshly
 * seeded transition. Only the r.done branch (and ensureRunAtom on the
 * next switch) writes those atoms.
 */
export const useVersionTransition = (): void => {
  const driver = useAtomValue(transitionDriverAtom);
  const size = useAtomValue(viewportSizeAtom);
  const setTransition = useSetAtom(transitionAtom);
  const setDriver = useSetAtom(transitionDriverAtom);
  const setViewport = useSetAtom(viewportAtom);
  const setSelection = useSetAtom(selectionAtom);
  const rafRef = useRef<number | null>(null);
  const sizeAtSeedRef = useRef(size);

  // If the viewport resizes mid-transition, the snapshots on driver.from/to
  // and the frozen driver.targetTransform are stretched/fit against the OLD
  // size, so layers would keep rendering stale geometry for the rest of the
  // ~4s window. Bail out: clearing transitionAtom hands layers back to
  // stretchedRunAtom at the new size, and viewportAtom=null lets
  // useFitOnActiveRun refit.
  useLayoutEffect(() => {
    if (!driver) {
      sizeAtSeedRef.current = size;
      return;
    }
    const seeded = sizeAtSeedRef.current;
    if (seeded.width === size.width && seeded.height === size.height) return;
    setTransition(null);
    setDriver(null);
    setViewport(null);
  }, [size, driver, setTransition, setDriver, setViewport]);

  useLayoutEffect(() => {
    if (!driver) return;
    // A version switch invalidates any open selection: the cluster /
    // creator detail belongs to the run we're leaving. Clear it only
    // when a switch actually starts (driver seeded), never on mount —
    // otherwise a hash-hydrated deep-link selection is wiped before it
    // can open.
    setSelection(null);
    const durations = buildDurations(driver.startTransform, driver.targetTransform);

    const tick = (now: number): void => {
      const elapsed = now - driver.startTime;
      const r = phaseAndProgress(elapsed, durations);

      if (r.phase === 0 && !r.done) {
        // Camera ease-out: fast start, gentle settle, matching the
        // tokens.motion.easeOut bezier used everywhere else.
        setViewport(lerpTransform(
          driver.startTransform,
          driver.targetTransform,
          easeOutCubic(r.progress),
        ));
      } else if (r.phase >= 1 && !r.done) {
        // Pin viewport at the phase-0 endpoint; pan/zoom listeners can
        // still write viewportAtom, so we re-pin per frame.
        setViewport(driver.targetTransform);
      }

      if (r.done) {
        setTransition(null);
        setDriver(null);
        setViewport(null);
        rafRef.current = null;
        return;
      }

      setTransition({
        from: driver.from,
        to: driver.to,
        phase: r.phase,
        progress: r.progress,
      });
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [driver, setTransition, setDriver, setViewport, setSelection]);
};
