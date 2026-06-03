import { atom, type Atom } from "jotai";
import type { ClusterLabel } from "@/data";
import { requireApiClient } from "./api-singleton";
import { activeRunIdAtom } from "./active-run-id.atom";
import { runStateAtom } from "./run.atom";

/**
 * Lazy per-cluster label (tags) cache, fetched on selection.
 *
 * The label block is the heavy part of a cluster's detail (~70% of the old
 * payload), so it is deferred out of the eager main-detail bundle and loaded
 * only when a cluster is opened. Entries are keyed `${runId}:${id}`; a cluster
 * with no label resolves to `null` (a real loaded value, distinct from "not yet
 * fetched"). While a fetch is in flight the inspector shows a tag skeleton.
 */
const keyFor = (runId: string, id: number): string => `${runId}:${id}`;

type LabelState = {
  // `has(key)` means loaded; the value is null for clusters with no label.
  byKey: Map<string, ClusterLabel | null>;
  loading: Set<string>;
  errors: Map<string, Error>;
};

export const clusterLabelMapAtom = atom<LabelState>({
  byKey: new Map<string, ClusterLabel | null>(),
  loading: new Set<string>(),
  errors: new Map<string, Error>(),
});

type Slot = {
  // present (possibly null) once loaded; absent while pending/erroring.
  label?: ClusterLabel | null;
  loading?: boolean;
  error?: Error;
};

const labelAtomCache = new Map<number, Atom<Slot>>();
export const clusterLabelFor = (id: number): Atom<Slot> => {
  const cached = labelAtomCache.get(id);
  if (cached) return cached;
  const a = atom<Slot>((get) => {
    const runId = get(activeRunIdAtom);
    if (!runId) return {};
    const k = keyFor(runId, id);
    const s = get(clusterLabelMapAtom);
    if (s.byKey.has(k)) return { label: s.byKey.get(k) ?? null };
    if (s.loading.has(k)) return { loading: true };
    const error = s.errors.get(k);
    if (error) return { error };
    return {};
  });
  labelAtomCache.set(id, a);
  return a;
};

/** Fetch one cluster's label on demand. Idempotent per (run, cluster). */
export const ensureClusterLabelAtom = atom(null, async (get, set, id: number) => {
  const runId = get(activeRunIdAtom);
  if (!runId) return;
  // The API client resolves the *committed* run; if the intended run (route) has
  // moved ahead of it (mid case-switch), fetching now would pull the old run's
  // label and cache it under the new run's key. Bail until they agree — the
  // ClusterPane effect re-fires on the committed-run change. Mirrors
  // ensureClusterBundleAtom.
  if (get(runStateAtom).activeRunId !== runId) return;
  const k = keyFor(runId, id);
  const s = get(clusterLabelMapAtom);
  if (s.byKey.has(k) || s.loading.has(k)) return;
  set(clusterLabelMapAtom, (prev) => {
    const loading = new Set(prev.loading);
    loading.add(k);
    const errors = new Map(prev.errors);
    errors.delete(k);
    return { ...prev, loading, errors };
  });
  try {
    const label = await requireApiClient().getClusterLabel(id);
    set(clusterLabelMapAtom, (prev) => {
      const byKey = new Map(prev.byKey).set(k, label);
      const loading = new Set(prev.loading);
      loading.delete(k);
      return { ...prev, byKey, loading };
    });
  } catch (err) {
    set(clusterLabelMapAtom, (prev) => {
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
