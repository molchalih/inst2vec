import { atom, type Atom } from "jotai";
import type { ClusterDetail } from "@/data";
import { requireApiClient } from "./api-singleton";
import { activeRunIdAtom } from "./active-run-id.atom";

// Cache entries are keyed by `${runId}:${id}` so that switching runs
// doesn't surface stale detail data for a colliding cluster id.
const keyFor = (runId: string, id: number): string => `${runId}:${id}`;

type Map_ = {
  details: Map<string, ClusterDetail>;
  loading: Set<string>;
  errors: Map<string, Error>;
};

export const clusterDetailMapAtom = atom<Map_>({
  details: new Map<string, ClusterDetail>(),
  loading: new Set<string>(),
  errors: new Map<string, Error>(),
});

type Slot = { data?: ClusterDetail; loading?: boolean; error?: Error };

// Memoized per-id atom: consumers call `clusterDetailFor(id)` on every
// render and must get back the same Atom instance. Returning a fresh
// derived atom each call would make `useAtomValue` re-subscribe and
// resnapshot every render — and since the derive function returns a
// brand-new Slot object each time, that resnapshot loops React.
// The runId is read inside the derive, so the same atom resolves to the
// right slot after a run switch without invalidating the cache.
const detailAtomCache = new Map<number, Atom<Slot>>();
export const clusterDetailFor = (id: number): Atom<Slot> => {
  const cached = detailAtomCache.get(id);
  if (cached) return cached;
  const a = atom<Slot>((get) => {
    const runId = get(activeRunIdAtom);
    if (!runId) return {};
    const k = keyFor(runId, id);
    const m = get(clusterDetailMapAtom);
    const data = m.details.get(k);
    if (data) return { data };
    if (m.loading.has(k)) return { loading: true };
    const error = m.errors.get(k);
    if (error) return { error };
    return {};
  });
  detailAtomCache.set(id, a);
  return a;
};

export const ensureClusterDetailAtom = atom(null, async (get, set, id: number) => {
  const runId = get(activeRunIdAtom);
  if (!runId) return;
  const k = keyFor(runId, id);
  const m = get(clusterDetailMapAtom);
  if (m.details.has(k) || m.loading.has(k)) return;
  set(clusterDetailMapAtom, (prev) => {
    const errors = new Map(prev.errors);
    errors.delete(k);
    const loading = new Set(prev.loading);
    loading.add(k);
    return { ...prev, loading, errors };
  });
  try {
    const detail = await requireApiClient().getClusterDetail(id);
    set(clusterDetailMapAtom, (prev) => {
      const details = new Map(prev.details).set(k, detail);
      const loading = new Set(prev.loading);
      loading.delete(k);
      return { ...prev, details, loading };
    });
  } catch (err) {
    set(clusterDetailMapAtom, (prev) => {
      const loading = new Set(prev.loading);
      loading.delete(k);
      const errors = new Map(prev.errors).set(
        k,
        err instanceof Error ? err : new Error(String(err)),
      );
      return { ...prev, loading, errors };
    });
    throw err;
  }
});
