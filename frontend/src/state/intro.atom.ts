import { atom, useAtomValue } from "jotai";
import type { Vec2, IntroPhase } from "@/core";

/**
 * Per-frame intro state, read by the dot + ellipse layers. centerWorld
 * is the world point that maps to screen-center under the fitted view;
 * it rides on the state (view data, scoped tightly — same pragmatic
 * exception as hover's screenX/Y) so the frame builders need no second
 * atom. Null when no intro is running.
 */
export type IntroState = {
  phase: IntroPhase;
  progress: number;
  centerWorld: Vec2;
} | null;

/**
 * rAF bookkeeping: written once when the intro is seeded, read only by
 * useIntroAnimation, cleared to null when the loop completes. Kept
 * separate from introAtom so per-frame phase/progress writes do not
 * invalidate consumers that only care that an intro exists.
 */
export type IntroDriver = {
  startTime: number;
  centerWorld: Vec2;
} | null;

export const introAtom = atom<IntroState>(null);
export const introDriverAtom = atom<IntroDriver>(null);

/** One-shot guard: flips true on the first load and never resets, so a
 *  later version switch or run change never re-triggers the intro. */
export const introPlayedAtom = atom<boolean>(false);

/**
 * True iff the one-time entrance flight is in flight. Derived so consumers
 * (version pills, drawer tongue) gate on one boolean that flips exactly
 * twice, instead of re-rendering on every per-frame phase/progress write.
 * Source of truth for "block interaction until the dots have settled".
 */
export const isIntroPlayingAtom = atom((get) => get(introAtom) !== null);

export const useIntro = (): IntroState => useAtomValue(introAtom);

export const useIsIntroPlaying = (): boolean => useAtomValue(isIntroPlayingAtom);

/** Whether the one-time entrance has been seeded (true for its whole life
 *  after first load, including the skip paths). Render layers read it to keep
 *  dots/ellipses hidden until the fade owns them — never flashing the static
 *  full-alpha frame in the gap between the run loading and the intro seeding. */
export const useIntroPlayed = (): boolean => useAtomValue(introPlayedAtom);

/**
 * Chrome entrance gate. True once the one-time intro flight has settled
 * (`introPlayed` flipped, no frame in flight) — also true immediately when
 * the intro is skipped for reduced-motion or a deep-link selection, since
 * `introPlayed` flips without a driver. The dock buttons and version drawer
 * stay absent until this turns true, then slide in. Stays a stable boolean
 * across the per-frame intro writes so consumers don't re-render mid-flight.
 */
export const chromeRevealedAtom = atom(
  (get) => get(introPlayedAtom) && get(introAtom) === null,
);

export const useIsChromeRevealed = (): boolean => useAtomValue(chromeRevealedAtom);
