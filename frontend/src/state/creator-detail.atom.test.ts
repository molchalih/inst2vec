import { describe, expect, it, beforeEach } from "vitest";
import { createStore } from "jotai";
import {
  creatorDetailMapAtom, ensureCreatorDetailAtom, creatorDetailFor,
} from "./creator-detail.atom";
import { manifestAtom } from "./manifest.atom";
import { runStateAtom } from "./run.atom";
import { selectionAtom } from "./selection.atom";
import { setApiClient } from "./api-singleton";
import type { ApiClient, AtlasRun, CreatorDetail, Manifest } from "@/data";
import { ApiUnavailableError, AssetNotFoundError } from "@/data";

const manifest = (defaultRunId: string, runIds: string[]): Manifest => ({
  version: 6,
  default_run_id: defaultRunId,
  runs: runIds.map((id) => ({
    id, case: "video", label: id, size: 1, details_available: true,
  })),
});

const seed = (runId: string) => {
  const store = createStore();
  store.set(manifestAtom, manifest(runId, [runId]));
  return store;
};

type Deferred<T> = { promise: Promise<T>; resolve: (v: T) => void; reject: (e: unknown) => void };
const defer = <T>(): Deferred<T> => {
  let resolve!: (v: T) => void; let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
};

const fakeDetail = (id: number): CreatorDetail => ({
  version: 6, user_id: id, cluster_id: 1, x: 0, y: 0, n_clips: 1,
  audio: { approachability: 0.5, engagement: 0.5, danceability: 0.5 },
  mood_shares: { happy: 0, sad: 0, relaxed: 0, aggressive: 0, party: 0 },
  timbre_shares: { acoustic: 0, electronic: 0, instrumental: 0, female_voice: 0, bright: 0, tonal: 0 },
  genre_top: [], instrument_top: [],
  speech: { detected_share: 0, top_langs: [] },
  caption: { top_langs: [] },
  posting: { median_plays: 0, median_clip_duration_s: 0, median_clips_per_week: 0, engagement_shape_ratio: 0 },
  follower_bucket: "1k", activity_span_months: 1,
  distinctiveness: [],
  spatial: {
    distance_from_centroid: 0,
    distance_from_centroid_percentile: 0,
    nearest_other_cluster: null,
  },
  clips: [],
});

class ProgrammableApi implements ApiClient {
  pending = new Map<number, Deferred<CreatorDetail>>();
  getCreatorDetail(id: number) {
    const d = defer<CreatorDetail>();
    this.pending.set(id, d);
    return d.promise;
  }
  getClusterDetail(): never { throw new Error("not used"); }
  searchCreators(): Promise<never> { return Promise.reject(new ApiUnavailableError("searchCreators")); }
  getEdges(): Promise<never> { return Promise.reject(new ApiUnavailableError("getEdges")); }
  getReels(): Promise<never> { return Promise.reject(new ApiUnavailableError("getReels")); }
  resolve(id: number) { this.pending.get(id)!.resolve(fakeDetail(id)); this.pending.delete(id); }
  fail(id: number, err: unknown) { this.pending.get(id)!.reject(err); this.pending.delete(id); }
}

describe("creator-detail.atom", () => {
  let api: ProgrammableApi;
  beforeEach(() => { api = new ProgrammableApi(); setApiClient(api); });

  it("loads, caches, and reports through creatorDetailFor", async () => {
    const store = seed("run-1");
    const p = store.set(ensureCreatorDetailAtom, 7);
    expect(store.get(creatorDetailFor(7))).toEqual({ loading: true });
    api.resolve(7);
    await p;
    expect(store.get(creatorDetailFor(7))).toEqual({
      data: expect.objectContaining({ user_id: 7 }),
    });
  });

  it("dedupes concurrent ensures for the same id", async () => {
    const store = seed("run-1");
    const p1 = store.set(ensureCreatorDetailAtom, 7);
    const p2 = store.set(ensureCreatorDetailAtom, 7);
    expect(api.pending.size).toBe(1);
    api.resolve(7);
    await Promise.all([p1, p2]);
    expect(store.get(creatorDetailMapAtom).details.has("run-1:7")).toBe(true);
  });

  it("records an error and clears loading on failure", async () => {
    const store = seed("run-1");
    const p = store.set(ensureCreatorDetailAtom, 7);
    api.fail(7, new Error("boom"));
    await p.catch(() => {});
    const s = store.get(creatorDetailFor(7));
    expect(s.error?.message).toBe("boom");
    expect(s.loading).toBeFalsy();
  });

  it("ensure after a prior failure retries the fetch", async () => {
    const store = seed("run-1");
    const p = store.set(ensureCreatorDetailAtom, 7);
    api.fail(7, new Error("boom"));
    await p.catch(() => {});
    expect(store.get(creatorDetailFor(7)).error).toBeTruthy();

    const p2 = store.set(ensureCreatorDetailAtom, 7);
    expect(store.get(creatorDetailFor(7)).loading).toBe(true);
    api.resolve(7);
    await p2;
    expect(store.get(creatorDetailFor(7)).data?.user_id).toBe(7);
    expect(store.get(creatorDetailFor(7)).error).toBeUndefined();
  });

  it("AssetNotFoundError → routes selection to the user's cluster and clears the slot", async () => {
    const store = seed("run-1");
    const run: AtlasRun = {
      meta: { id: "run-1", case: "video", label: "v", size: 1, details_available: true },
      bounds: { minX: -1, maxX: 1, minY: -1, maxY: 1 },
      users: [[7, 0, 0, 4, true, 0.5]],
      clusters: [],
    };
    store.set(runStateAtom, { runs: new Map([["run-1", run]]), activeRunId: "run-1" });

    const p = store.set(ensureCreatorDetailAtom, 7);
    api.fail(7, new AssetNotFoundError("/data/runs/run-1/users/7.json"));
    await p;

    expect(store.get(creatorDetailFor(7))).toEqual({});
    expect(store.get(selectionAtom)).toEqual({ kind: "cluster", clusterId: 4 });
  });

  it("AssetNotFoundError with no resolvable user → rethrows without selection change", async () => {
    const store = seed("run-1");
    // No runState seeded → activeRunAtom resolves to null, no fallback target.
    const p = store.set(ensureCreatorDetailAtom, 99);
    api.fail(99, new AssetNotFoundError("/data/runs/run-1/users/99.json"));
    await expect(p).rejects.toBeInstanceOf(AssetNotFoundError);
    expect(store.get(selectionAtom)).toBeNull();
  });

  it("refetches when the active run changes for the same id", async () => {
    const store = createStore();
    store.set(manifestAtom, manifest("run-a", ["run-a"]));
    const p1 = store.set(ensureCreatorDetailAtom, 7);
    api.resolve(7);
    await p1;
    expect(store.get(creatorDetailFor(7)).data?.user_id).toBe(7);

    store.set(manifestAtom, manifest("run-b", ["run-b"]));
    expect(store.get(creatorDetailFor(7))).toEqual({});
    const p2 = store.set(ensureCreatorDetailAtom, 7);
    expect(api.pending.size).toBe(1);
    api.resolve(7);
    await p2;
    expect(store.get(creatorDetailMapAtom).details.has("run-a:7")).toBe(true);
    expect(store.get(creatorDetailMapAtom).details.has("run-b:7")).toBe(true);
  });
});
