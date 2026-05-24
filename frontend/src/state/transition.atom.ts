import { atom, useAtomValue } from "jotai";
import type { AtlasRun } from "@/data";
import type { Transform } from "@/core";

export type TransitionPhase = 0 | 1 | 2 | 3;

/**
 * Per-switch durations the rAF loop reads. Mirrors the 0-3 internal
 * state-machine indexing of phaseAndProgress, not the human-facing
 * phase1-4 token names. Computed once at seed time inside
 * ensureRunAtom (where phase0 can be zeroed when the camera doesn't
 * need to move) and frozen on the driver for the lifetime of the run.
 */
export type PhaseDurations = {
  phase0: number; phase1: number; phase2: number; phase3: number;
};

export type TransitionState = {
  from: AtlasRun;
  to: AtlasRun;
  phase: TransitionPhase;
  progress: number;
};

/**
 * Bookkeeping for the rAF loop: written once when a transition is seeded,
 * read only by useVersionTransition, cleared back to null when the loop
 * completes. Kept separate from transitionAtom so per-frame phase/progress
 * updates do not invalidate consumers that only care about the runs in flight.
 */
export type TransitionDriver = {
  from: AtlasRun;
  to: AtlasRun;
  startTransform: Transform;
  targetTransform: Transform;
  startTime: number;
};

export const transitionAtom = atom<TransitionState | null>(null);
export const transitionDriverAtom = atom<TransitionDriver | null>(null);

/**
 * True iff a version-switch transition is currently in flight. Derived
 * so consumers (pills, command palette, deep-link router) can read one
 * boolean instead of inspecting transitionAtom. Source of truth for
 * "block further switches" UX and writer-side guards.
 */
export const isTransitioningAtom = atom((get) => get(transitionAtom) !== null);

export const useTransition = (): TransitionState | null =>
  useAtomValue(transitionAtom);

export const useIsTransitioning = (): boolean =>
  useAtomValue(isTransitioningAtom);
