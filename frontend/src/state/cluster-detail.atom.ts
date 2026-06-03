import { atom, type Atom } from "jotai";
import type { ClusterDetail } from "@/data";
import { requireApiClient } from "./api-singleton";
import { activeRunIdAtom } from "./active-run-id.atom";
import { runStateAtom } from "./run.atom";

/**
 * Eager cluster main-detail cache, one entry per run.
 *
 * The per-run `clusters-detail.json` bundle is fetched once (on run load and on
 * pill switch) and cached keyed by runId. Switching runs never surfaces stale
 * detail for a colliding cluster id because the cache is run-scoped. The heavy
 * label/tags live in a separate per-cluster file — see `cluster-label.atom`.
 */
type BundleState = {
  byRun: Map<string, Map<number, ClusterDetail>>;
  loading: Set<string>;
  errors: Map<string, Error>;
};

export const clusterDetailBundleAtom = atom<BundleState>({
  byRun: new Map<string, Map<number, ClusterDetail>>(),
  loading: new Set<string>(),
  errors: new Map<string, Error>(),
});

type Slot = { data?: ClusterDetail; loading?: boolean; error?: Error };

// Memoized per-id selector atom (same instance per id across renders, mirroring
// the creator-detail pattern): the runId is read inside the derive so the same
// atom resolves to the right run's bundle after a switch.
const detailAtomCache = new Map<number, Atom<Slot>>();
export const clusterDetailFor = (id: number): Atom<Slot> => {
  const cached = detailAtomCache.get(id);
  if (cached) return cached;
  const a = atom<Slot>((get) => {
    const runId = get(activeRunIdAtom);
    if (!runId) return {};
    const s = get(clusterDetailBundleAtom);
    const m = s.byRun.get(runId);
    if (m) {
      const data = m.get(id);
      return data ? { data } : {};
    }
    if (s.loading.has(runId)) return { loading: true };
    const error = s.errors.get(runId);
    if (error) return { error };
    return {};
  });
  detailAtomCache.set(id, a);
  return a;
};

/**
 * Fetch the active run's main-detail bundle into the cache if absent. Idempotent
 * (a cache hit or in-flight request is a no-op). Driven eagerly by the app-level
 * prefetch so detail is present the instant a cluster is clicked.
 */
export const ensureClusterBundleAtom = atom(null, async (get, set) => {
  const runId = get(activeRunIdAtom);
  if (!runId) return;
  // The API client resolves the *committed* run (`runStateAtom.activeRunId`),
  // which lags `activeRunIdAtom` until the bulk run loads. Fetching in that
  // window throws "no active runId" and caches the error. Bail until the
  // committed run has caught up so the fetch always targets a live run.
  if (get(runStateAtom).activeRunId !== runId) return;
  const s = get(clusterDetailBundleAtom);
  if (s.byRun.has(runId) || s.loading.has(runId)) return;
  set(clusterDetailBundleAtom, (prev) => {
    const loading = new Set(prev.loading);
    loading.add(runId);
    const errors = new Map(prev.errors);
    errors.delete(runId);
    return { ...prev, loading, errors };
  });
  try {
    const clusters = await requireApiClient().getClustersDetail();
    set(clusterDetailBundleAtom, (prev) => {
      const byRun = new Map(prev.byRun);
      byRun.set(runId, new Map(clusters.map((c) => [c.cluster_id, c])));
      const loading = new Set(prev.loading);
      loading.delete(runId);
      return { ...prev, byRun, loading };
    });
  } catch (err) {
    set(clusterDetailBundleAtom, (prev) => {
      const loading = new Set(prev.loading);
      loading.delete(runId);
      const errors = new Map(prev.errors).set(
        runId,
        err instanceof Error ? err : new Error(String(err)),
      );
      return { ...prev, loading, errors };
    });
    throw err;
  }
});
