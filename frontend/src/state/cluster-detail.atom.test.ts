import { describe, expect, it, beforeEach } from "vitest";
import { createStore } from "jotai";
import {
  clusterDetailBundleAtom, ensureClusterBundleAtom, clusterDetailFor,
} from "./cluster-detail.atom";
import { manifestAtom } from "./manifest.atom";
import { runStateAtom } from "./run.atom";
import { setApiClient } from "./api-singleton";
import type { ApiClient, ClusterDetail, ClusterLabel, Manifest } from "@/data";
import { ApiUnavailableError } from "@/data";

const manifest = (defaultRunId: string, runIds: string[]): Manifest => ({
  version: 7,
  default_run_id: defaultRunId,
  runs: runIds.map((id) => ({
    id, case: "video", label: id, size: 1, details_available: true,
  })),
});

// Seed both the intended run (manifest → activeRunIdAtom) and the committed
// run (runStateAtom.activeRunId). The bundle only fetches once the run is
// committed, because the API client resolves the committed run id.
const seed = (runId: string) => {
  const store = createStore();
  store.set(manifestAtom, manifest(runId, [runId]));
  store.set(runStateAtom, { runs: new Map(), activeRunId: runId });
  return store;
};

type Deferred<T> = { promise: Promise<T>; resolve: (v: T) => void; reject: (e: unknown) => void };
const defer = <T>(): Deferred<T> => {
  let resolve!: (v: T) => void; let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
};

const fakeDetail = (id: number): ClusterDetail => ({
  cluster_id: id, size: 1,
  ellipse: { cx: 0, cy: 0, rx: 1, ry: 1, angle: 0 },
  audio: { approachability: 0.5, engagement: 0.5, danceability: 0.5 },
  mood_shares: { happy: 0, sad: 0, relaxed: 0, aggressive: 0, party: 0 },
  timbre_shares: { acoustic: 0, electronic: 0, instrumental: 0, female_voice: 0, bright: 0, tonal: 0 },
  genre_top: [], instrument_top: [],
  speech: { detected_share: 0, top_langs: [] },
  caption: { top_langs: [] },
  posting: { median_plays: 0, median_clip_duration_s: 0, median_clips_per_week: 0, engagement_shape_ratio: 0 },
  follower_bucket: "1k", activity_span_months: 1,
  distinctiveness: [],
  spatial: { compactness: 0, nearest_clusters: [] },
  label_modality: null,
});

// One pending bundle request at a time (the prefetch is per-run, no id).
class ProgrammableApi implements ApiClient {
  pending: Deferred<ClusterDetail[]> | null = null;
  callCount = 0;
  getClustersDetail() {
    this.callCount += 1;
    const d = defer<ClusterDetail[]>();
    this.pending = d;
    return d.promise;
  }
  getClusterLabel(): Promise<ClusterLabel | null> { return Promise.resolve(null); }
  getCreatorDetail(): never { throw new Error("not used"); }
  searchCreators(): Promise<never> { return Promise.reject(new ApiUnavailableError("searchCreators")); }
  getEdges(): Promise<never> { return Promise.reject(new ApiUnavailableError("getEdges")); }
  getReels(): Promise<never> { return Promise.reject(new ApiUnavailableError("getReels")); }
  resolve(ids: number[]) { this.pending!.resolve(ids.map(fakeDetail)); this.pending = null; }
  fail(err: unknown) { this.pending!.reject(err); this.pending = null; }
}

describe("cluster-detail.atom (bundle)", () => {
  let api: ProgrammableApi;
  beforeEach(() => { api = new ProgrammableApi(); setApiClient(api); });

  it("loads the bundle and reports through clusterDetailFor", async () => {
    const store = seed("run-1");
    const p = store.set(ensureClusterBundleAtom);
    expect(store.get(clusterDetailFor(7))).toEqual({ loading: true });
    api.resolve([7, 8]);
    await p;
    expect(store.get(clusterDetailFor(7))).toEqual({
      data: expect.objectContaining({ cluster_id: 7 }),
    });
    // A cluster absent from the bundle resolves to no-data (not loading).
    expect(store.get(clusterDetailFor(999))).toEqual({});
  });

  it("dedupes concurrent ensures for the same run", async () => {
    const store = seed("run-1");
    const p1 = store.set(ensureClusterBundleAtom);
    const p2 = store.set(ensureClusterBundleAtom);
    expect(api.callCount).toBe(1);
    api.resolve([7]);
    await Promise.all([p1, p2]);
    expect(store.get(clusterDetailBundleAtom).byRun.has("run-1")).toBe(true);
  });

  it("records an error and clears loading on failure", async () => {
    const store = seed("run-1");
    const p = store.set(ensureClusterBundleAtom);
    api.fail(new Error("boom"));
    await p.catch(() => {});
    const s = store.get(clusterDetailFor(7));
    expect(s.error?.message).toBe("boom");
    expect(s.loading).toBeFalsy();
  });

  it("ensure after a prior failure retries the fetch", async () => {
    const store = seed("run-1");
    const p = store.set(ensureClusterBundleAtom);
    api.fail(new Error("boom"));
    await p.catch(() => {});
    expect(store.get(clusterDetailFor(7)).error).toBeTruthy();

    const p2 = store.set(ensureClusterBundleAtom);
    expect(store.get(clusterDetailFor(7)).loading).toBe(true);
    api.resolve([7]);
    await p2;
    expect(store.get(clusterDetailFor(7)).data?.cluster_id).toBe(7);
    expect(store.get(clusterDetailFor(7)).error).toBeUndefined();
  });

  it("refetches when the active run changes", async () => {
    const store = createStore();
    store.set(manifestAtom, manifest("run-a", ["run-a"]));
    store.set(runStateAtom, { runs: new Map(), activeRunId: "run-a" });
    const p1 = store.set(ensureClusterBundleAtom);
    api.resolve([7]);
    await p1;
    expect(store.get(clusterDetailFor(7)).data?.cluster_id).toBe(7);

    store.set(manifestAtom, manifest("run-b", ["run-b"]));
    store.set(runStateAtom, { runs: new Map(), activeRunId: "run-b" });
    expect(store.get(clusterDetailFor(7))).toEqual({});
    const p2 = store.set(ensureClusterBundleAtom);
    api.resolve([7]);
    await p2;
    expect(store.get(clusterDetailBundleAtom).byRun.has("run-a")).toBe(true);
    expect(store.get(clusterDetailBundleAtom).byRun.has("run-b")).toBe(true);
  });

  // Regression: the eager prefetch can fire the instant the manifest resolves
  // the intended run id — but the API client resolves the *committed* run
  // (runStateAtom.activeRunId), which lags until the bulk run loads. Fetching in
  // that window threw "no active runId" and cached the error ("Couldn't load
  // detail"). ensureClusterBundle must no-op until the committed run is ready.
  it("no-ops until the committed run matches the intended run", async () => {
    const store = createStore();
    store.set(manifestAtom, manifest("run-1", ["run-1"])); // intended set
    // committed still null (bulk run not loaded yet)
    await store.set(ensureClusterBundleAtom);
    expect(api.callCount).toBe(0);
    expect(store.get(clusterDetailFor(7))).toEqual({});

    store.set(runStateAtom, { runs: new Map(), activeRunId: "run-1" }); // commit
    const p = store.set(ensureClusterBundleAtom);
    expect(api.callCount).toBe(1);
    api.resolve([7]);
    await p;
    expect(store.get(clusterDetailFor(7)).data?.cluster_id).toBe(7);
  });
});
