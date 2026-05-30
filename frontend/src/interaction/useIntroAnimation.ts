import { useLayoutEffect, useRef } from "react";
import { useAtomValue, useSetAtom, useStore } from "jotai";
import {
  stretchedRunAtom, viewportAtom, viewportSizeAtom,
  introAtom, introDriverAtom, introPlayedAtom,
  parseHash,
} from "@/state";
import { screenToWorld, introPhaseAndProgress } from "@/core";
import { tokens } from "@/ui/tokens";

const prefersReducedMotion = (): boolean =>
  typeof globalThis !== "undefined"
  && globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true;

const hasDeepLinkSelection = (): boolean => {
  if (typeof globalThis === "undefined") return false;
  const parsed = parseHash(globalThis.location.hash);
  return parsed.user !== undefined || parsed.cluster !== undefined;
};

/**
 * One-time page-load entrance. Seeds the intro driver when the first
 * run is ready (skipping for reduced-motion or a deep-link selection),
 * then drives fade → flight → settle off tokens.motion.intro.
 *
 * Seed and drive are split (mirroring useVersionTransition): the seed
 * effect is guarded by introPlayedAtom so it never fires twice; the
 * drive effect consumes the driver atom, so a strict-mode remount
 * re-runs it against the same persisted driver and restarts the rAF
 * from driver.startTime rather than stranding the animation.
 */
export const useIntroAnimation = (): void => {
  const run = useAtomValue(stretchedRunAtom);
  const size = useAtomValue(viewportSizeAtom);
  const driver = useAtomValue(introDriverAtom);
  const setIntro = useSetAtom(introAtom);
  const setDriver = useSetAtom(introDriverAtom);
  const setPlayed = useSetAtom(introPlayedAtom);
  const store = useStore();
  const rafRef = useRef<number | null>(null);
  const sizeAtSeedRef = useRef(size);

  // Seed once: run ready + size measured + not already played.
  useLayoutEffect(() => {
    if (!run) return;
    if (size.width <= 0 || size.height <= 0) return;
    if (store.get(introPlayedAtom)) return;
    setPlayed(true);
    if (prefersReducedMotion() || hasDeepLinkSelection()) return;

    // Camera is the derived fit on first load (no override set yet), so
    // viewportAtom already reads the fitted transform.
    const fitted = store.get(viewportAtom);
    const centerWorld = screenToWorld(
      { x: size.width / 2, y: size.height / 2 },
      fitted,
    );
    sizeAtSeedRef.current = size;
    setDriver({ startTime: performance.now(), centerWorld });
  }, [run, size, store, setPlayed, setDriver]);

  // Bail on mid-intro resize: the seeded centerWorld and the run's
  // stretched targets are stale at the new size. Clearing hands the
  // layers back to the static frame at the refit.
  useLayoutEffect(() => {
    if (!driver) { sizeAtSeedRef.current = size; return; }
    const seeded = sizeAtSeedRef.current;
    if (seeded.width === size.width && seeded.height === size.height) return;
    setIntro(null);
    setDriver(null);
  }, [size, driver, setIntro, setDriver]);

  // Drive the rAF timeline whenever a driver exists.
  useLayoutEffect(() => {
    if (!driver) return;
    const durations = {
      fadeMs: tokens.motion.intro.fadeMs,
      flightMs: tokens.motion.intro.flightMs,
      settleMs: tokens.motion.intro.settleMs,
    };

    const tick = (now: number): void => {
      const elapsed = now - driver.startTime;
      const r = introPhaseAndProgress(elapsed, durations);
      if (r.done) {
        setIntro(null);
        setDriver(null);
        rafRef.current = null;
        return;
      }
      setIntro({ phase: r.phase, progress: r.progress, centerWorld: driver.centerWorld });
      rafRef.current = requestAnimationFrame(tick);
    };

    // Paint the first stacked-at-center frame synchronously so dots
    // never flash at their final positions before the first rAF.
    setIntro({ phase: 0, progress: 0, centerWorld: driver.centerWorld });
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [driver, setIntro, setDriver]);
};
