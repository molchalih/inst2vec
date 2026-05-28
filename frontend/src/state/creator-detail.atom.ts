import { atom, type Atom } from "jotai";
import { AssetNotFoundError, type CreatorDetail } from "@/data";
import { requireApiClient } from "./api-singleton";
import { activeRunIdAtom } from "./active-run-id.atom";
import { activeRunAtom } from "./run.atom";
import { selectionAtom } from "./selection.atom";

// Cache entries are keyed by `${runId}:${id}` so that switching runs
// doesn't surface stale detail data for a colliding creator id.
type Key = string;
const keyFor = (runId: string, id: number): Key => `${runId}:${id}`;

type Map_ = {
  details: Map<Key, CreatorDetail>;
  loading: Set<Key>;
  errors: Map<Key, Error>;
};

export const creatorDetailMapAtom = atom<Map_>({
  details: new Map<Key, CreatorDetail>(),
  loading: new Set<Key>(),
  errors: new Map<Key, Error>(),
});

type Slot = { data?: CreatorDetail; loading?: boolean; error?: Error };

// Memoized per-id atom: consumers call `creatorDetailFor(id)` on every
// render and must get back the same Atom instance. Returning a fresh
// derived atom each call would make `useAtomValue` re-subscribe and
// resnapshot every render — and since the derive function returns a
// brand-new Slot object each time, that resnapshot loops React.
// The runId is read inside the derive, so the same atom resolves to the
// right slot after a run switch without invalidating the cache.
const detailAtomCache = new Map<number, Atom<Slot>>();
export const creatorDetailFor = (id: number): Atom<Slot> => {
  const cached = detailAtomCache.get(id);
  if (cached) return cached;
  const a = atom<Slot>((get) => {
    const runId = get(activeRunIdAtom);
    if (!runId) return {};
    const k = keyFor(runId, id);
    const m = get(creatorDetailMapAtom);
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

export const ensureCreatorDetailAtom = atom(null, async (get, set, id: number) => {
  const runId = get(activeRunIdAtom);
  if (!runId) return;
  const k = keyFor(runId, id);
  const m = get(creatorDetailMapAtom);
  if (m.details.has(k) || m.loading.has(k)) return;
  set(creatorDetailMapAtom, (prev) => {
    const errors = new Map(prev.errors);
    errors.delete(k);
    const loading = new Set(prev.loading);
    loading.add(k);
    return { ...prev, loading, errors };
  });
  try {
    const detail = await requireApiClient().getCreatorDetail(id);
    set(creatorDetailMapAtom, (prev) => {
      const details = new Map(prev.details).set(k, detail);
      const loading = new Set(prev.loading);
      loading.delete(k);
      return { ...prev, details, loading };
    });
  } catch (err) {
    set(creatorDetailMapAtom, (prev) => {
      const loading = new Set(prev.loading);
      loading.delete(k);
      if (err instanceof AssetNotFoundError) {
        // Missing per-creator JSON: don't record an error — the
        // fallback below transitions selection to the cluster pane.
        return { ...prev, loading };
      }
      const errors = new Map(prev.errors).set(
        k,
        err instanceof Error ? err : new Error(String(err)),
      );
      return { ...prev, loading, errors };
    });
    if (err instanceof AssetNotFoundError) {
      // Manifest may advertise details_available=true while the per-id
      // JSON was never exported (committed assets, fresh-clone builds).
      // Route the selection to the user's cluster instead of surfacing
      // a fetch-error UI for data that will never arrive.
      const run = get(activeRunAtom);
      const user = run?.users.find(([uid]) => uid === id);
      const clusterId = user?.[3];
      if (clusterId !== undefined && clusterId >= 0) {
        set(selectionAtom, { kind: "cluster", clusterId });
        return;
      }
    }
    throw err;
  }
});
