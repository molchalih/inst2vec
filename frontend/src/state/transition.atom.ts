import { atom, useAtomValue } from "jotai";
import { selectAtom } from "jotai/utils";
import {
  joinUsersByCreator,
  type JoinedUser,
  type Transform,
} from "@/core";
import type { AtlasRun } from "@/data";

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

/**
 * The two runs in flight, identity-stable across per-frame phase/progress ticks.
 * selectAtom only emits a new value when the run pair actually changes, so the
 * derived join below rebuilds once per switch, not every rAF frame.
 */
const morphRunsAtom = selectAtom(
  transitionAtom,
  (t): { from: AtlasRun; to: AtlasRun } | null =>
    t ? { from: t.from, to: t.to } : null,
  (a, b) => a?.from === b?.from && a?.to === b?.to,
);

/**
 * Creator-keyed join of the two morphing runs, shared by DotsLayer and
 * TrackingLayer so neither recomputes it and the halo travels on exactly the
 * same per-creator interpolation the dots use. Null when no switch is in flight.
 */
export const morphJoinAtom = atom<ReadonlyArray<JoinedUser> | null>((get) => {
  const runs = get(morphRunsAtom);
  return runs ? joinUsersByCreator(runs.from, runs.to) : null;
});

export const useTransition = (): TransitionState | null =>
  useAtomValue(transitionAtom);

export const useMorphJoin = (): ReadonlyArray<JoinedUser> | null =>
  useAtomValue(morphJoinAtom);

export const useIsTransitioning = (): boolean =>
  useAtomValue(isTransitioningAtom);
