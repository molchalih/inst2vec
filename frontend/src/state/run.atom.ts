import { atom, useAtomValue } from "jotai";
import type { AtlasRun } from "@/data";
import { fitBounds } from "@/core";
import { requireBulkSource } from "./bulk-singleton";
import { stretchedRunAtom } from "./stretched-run.atom";
import { viewportSizeAtom } from "./viewport-size.atom";
import { viewportAtom } from "./viewport.atom";
import { hoverAtom } from "./hover.atom";
import { transitionAtom, transitionDriverAtom } from "./transition.atom";

export type RunState = {
  runs: Map<string, AtlasRun>;
  activeRunId: string | null;
};

export const runStateAtom = atom<RunState>({ runs: new Map(), activeRunId: null });

export const activeRunAtom = atom<AtlasRun | null>((get) => {
  const s = get(runStateAtom);
  return s.activeRunId ? s.runs.get(s.activeRunId) ?? null : null;
});

// Latest runId the app has asked for. Written at the top of every
// ensureRunAtom call; the post-await activate step bails out if it
// no longer matches, so a slow earlier fetch cannot clobber the
// activeRunId selected by a newer request.
export const requestedRunIdAtom = atom<string | null>(null);

export const ensureRunAtom = atom(null, async (get, set, runId: string) => {
  // Writer-side guard: a transition in flight blocks further switches.
  // The VersionPill is also disabled via useIsTransitioning, so a normal
  // user can't reach this path; the guard exists as defence-in-depth for
  // any other caller (deep-link router, programmatic switch, etc.).
  if (get(transitionAtom) !== null) return;

  const bulk = requireBulkSource();
  set(requestedRunIdAtom, runId);

  if (!get(runStateAtom).runs.has(runId)) {
    const run = await bulk.getRun(runId);
    set(runStateAtom, (prev) => ({
      ...prev,
      runs: new Map(prev.runs).set(runId, run),
    }));
  }

  if (get(requestedRunIdAtom) !== runId) return;

  // Capture pre-flip stretched run + viewport in the same synchronous batch as
  // the activeRunId flip + transition seed. useVersionTransition consumes
  // transitionDriverAtom as a pure rAF driver; doing this here (vs in a
  // useEffect downstream of stretchedRunAtom) closes the one-frame window
  // where the destination atlas would paint with transitionAtom still null.
  const fromStretched = get(stretchedRunAtom);
  const startTransform = get(viewportAtom);

  set(runStateAtom, (prev) => ({ ...prev, activeRunId: runId }));

  const toStretched = get(stretchedRunAtom);
  if (
    fromStretched && toStretched
    && fromStretched.meta.id !== toStretched.meta.id
  ) {
    const size = get(viewportSizeAtom);
    const targetTransform = fitBounds(fromStretched.bounds, size);
    set(hoverAtom, { dotId: null, clusterId: null, screenX: 0, screenY: 0 });
    // Pin viewport at the user's pre-switch transform so the first paint
    // matches what the rAF will write at progress=0. Writing targetTransform
    // here would snap to fit(OLD) for one frame before the rAF lerps back
    // toward startTransform — the same class of one-frame flash this commit
    // exists to eliminate. The phase-0 lerp inside useVersionTransition drives
    // viewport to targetTransform over phase0 ms.
    set(viewportAtom, startTransform);
    set(transitionAtom, {
      from: fromStretched,
      to: toStretched,
      phase: 0,
      progress: 0,
    });
    set(transitionDriverAtom, {
      from: fromStretched,
      to: toStretched,
      startTransform,
      targetTransform,
      startTime: performance.now(),
    });
  }
});

/**
 * Cache-only run loader — the load half of ensureRunAtom with the
 * activate/transition half removed. Fetches a run and inserts it into the
 * cache WITHOUT touching activeRunId, requestedRunIdAtom, transitionAtom, or
 * viewportAtom, so an eager prefetch (app/TrackingPrefetch) can fill the cache
 * the presence gate reads without driving a version switch. Idempotent: a
 * cache hit is a no-op.
 */
export const prefetchRunAtom = atom(null, async (get, set, runId: string) => {
  if (get(runStateAtom).runs.has(runId)) return;
  const bulk = requireBulkSource();
  const run = await bulk.getRun(runId);
  set(runStateAtom, (prev) => {
    if (prev.runs.has(runId)) return prev;
    return { ...prev, runs: new Map(prev.runs).set(runId, run) };
  });
});

export const useActiveRun = () => useAtomValue(activeRunAtom);
