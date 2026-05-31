import { atom, useAtomValue } from "jotai";
import { isCreatorInRun } from "@/core";
import { trackedCreatorAtom } from "./tracking.atom";
import { runStateAtom } from "./run.atom";
import { manifestAtom } from "./manifest.atom";

/**
 * Per-run presence of the tracked creator, keyed by run id (`meta.id`).
 *
 * Derived (read-only). When nothing is tracked the gate is inert — every run
 * reports present, so no pill is disabled. When a creator is tracked, presence
 * is computed exactly from the cached run via `isCreatorInRun`; a run not yet
 * in the cache is treated as *absent* (pessimistic). This preserves the hard
 * presence invariant (§2.4): a visitor can never switch to a case lacking the
 * tracked creator, so during the prefetch window an unproven case must stay
 * disabled rather than briefly clickable. The eager prefetch
 * (app/TrackingPrefetch) loads every run shortly after the first track, filling
 * the cache so pills enable for the cases that actually contain the creator.
 * (The active run is always cached and contains the tracked creator — tracking
 * can only start by clicking a dot in the active run — so the active pill is
 * never wrongly disabled.)
 */
export const trackedPresenceAtom = atom<Record<string, boolean>>((get) => {
  const tracked = get(trackedCreatorAtom);
  const manifest = get(manifestAtom);
  const map: Record<string, boolean> = {};
  if (!manifest) return map;
  if (tracked === null) {
    for (const run of manifest.runs) map[run.id] = true;
    return map;
  }
  const cache = get(runStateAtom).runs;
  for (const run of manifest.runs) {
    const cached = cache.get(run.id);
    map[run.id] = cached ? isCreatorInRun(cached, tracked) : false;
  }
  return map;
});

export const useTrackedPresentInRun = (runId: string): boolean =>
  useAtomValue(trackedPresenceAtom)[runId] ?? true;
