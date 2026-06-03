import { describe, expect, it, beforeEach } from "vitest";
import { createStore } from "jotai";
import {
  clusterLabelMapAtom, ensureClusterLabelAtom, clusterLabelFor,
} from "./cluster-label.atom";
import { manifestAtom } from "./manifest.atom";
import { runStateAtom } from "./run.atom";
import { setApiClient } from "./api-singleton";
import type { ApiClient, ClusterLabel, Manifest } from "@/data";
import { ApiUnavailableError } from "@/data";

const manifest = (defaultRunId: string, runIds: string[]): Manifest => ({
  version: 7,
  default_run_id: defaultRunId,
  runs: runIds.map((id) => ({
    id, case: "video", label: id, size: 1, details_available: true,
  })),
});

// Seed intended (manifest → activeRunIdAtom) and committed (runStateAtom);
// the label fetch only proceeds once the committed run matches the intended one.
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

const fakeLabel = (name: string): ClusterLabel => ({
  label: name, summary: "s", modality: "visual",
  repertoire: [], aesthetic_logic: [],
  taste_signalling: { label: "t", description: "d", confidence: "medium" },
  visibility_orientation: { label: "v", description: "d", confidence: "low" },
  internal_variations: [], boundary_notes: "", tool_tags: [],
  validation: "ok", warnings: [],
});

class ProgrammableApi implements ApiClient {
  pending = new Map<number, Deferred<ClusterLabel | null>>();
  getClustersDetail(): Promise<never[]> { return Promise.resolve([]); }
  getClusterLabel(id: number) {
    const d = defer<ClusterLabel | null>();
    this.pending.set(id, d);
    return d.promise;
  }
  getCreatorDetail(): never { throw new Error("not used"); }
  searchCreators(): Promise<never> { return Promise.reject(new ApiUnavailableError("searchCreators")); }
  getEdges(): Promise<never> { return Promise.reject(new ApiUnavailableError("getEdges")); }
  getReels(): Promise<never> { return Promise.reject(new ApiUnavailableError("getReels")); }
  resolve(id: number, label: ClusterLabel | null) { this.pending.get(id)!.resolve(label); this.pending.delete(id); }
  fail(id: number, err: unknown) { this.pending.get(id)!.reject(err); this.pending.delete(id); }
}

describe("cluster-label.atom (lazy tags)", () => {
  let api: ProgrammableApi;
  beforeEach(() => { api = new ProgrammableApi(); setApiClient(api); });

  it("reports loading then the loaded label", async () => {
    const store = seed("run-1");
    const p = store.set(ensureClusterLabelAtom, 7);
    expect(store.get(clusterLabelFor(7))).toEqual({ loading: true });
    api.resolve(7, fakeLabel("Cinematic Synth"));
    await p;
    expect(store.get(clusterLabelFor(7)).label?.label).toBe("Cinematic Synth");
  });

  it("treats a null label as loaded (cluster with no tags)", async () => {
    const store = seed("run-1");
    const p = store.set(ensureClusterLabelAtom, 7);
    api.resolve(7, null);
    await p;
    // Loaded, value null — distinct from the not-yet-fetched {} state.
    expect(store.get(clusterLabelFor(7))).toEqual({ label: null });
  });

  it("dedupes concurrent ensures for the same id", async () => {
    const store = seed("run-1");
    const p1 = store.set(ensureClusterLabelAtom, 7);
    const p2 = store.set(ensureClusterLabelAtom, 7);
    expect(api.pending.size).toBe(1);
    api.resolve(7, null);
    await Promise.all([p1, p2]);
    expect(store.get(clusterLabelMapAtom).byKey.has("run-1:7")).toBe(true);
  });

  it("records an error and clears loading on failure", async () => {
    const store = seed("run-1");
    const p = store.set(ensureClusterLabelAtom, 7);
    api.fail(7, new Error("boom"));
    await p.catch(() => {});
    const s = store.get(clusterLabelFor(7));
    expect(s.error?.message).toBe("boom");
    expect(s.loading).toBeFalsy();
  });

  it("refetches when the active run changes for the same id", async () => {
    const store = createStore();
    store.set(manifestAtom, manifest("run-a", ["run-a"]));
    store.set(runStateAtom, { runs: new Map(), activeRunId: "run-a" });
    const p1 = store.set(ensureClusterLabelAtom, 7);
    api.resolve(7, fakeLabel("A"));
    await p1;
    expect(store.get(clusterLabelFor(7)).label?.label).toBe("A");

    store.set(manifestAtom, manifest("run-b", ["run-b"]));
    store.set(runStateAtom, { runs: new Map(), activeRunId: "run-b" });
    expect(store.get(clusterLabelFor(7))).toEqual({});
    const p2 = store.set(ensureClusterLabelAtom, 7);
    api.resolve(7, fakeLabel("B"));
    await p2;
    expect(store.get(clusterLabelFor(7)).label?.label).toBe("B");
  });

  // Regression (codex P2): the label fetch must not run while the route's
  // intended run is ahead of the committed run, or it would cache the old run's
  // label under the new run's key.
  it("no-ops until the committed run matches the intended run", async () => {
    const store = createStore();
    store.set(manifestAtom, manifest("run-1", ["run-1"])); // intended
    // committed still on the previous run (mid case-switch)
    store.set(runStateAtom, { runs: new Map(), activeRunId: "run-0" });
    await store.set(ensureClusterLabelAtom, 7);
    expect(api.pending.size).toBe(0);
    expect(store.get(clusterLabelFor(7))).toEqual({});

    store.set(runStateAtom, { runs: new Map(), activeRunId: "run-1" }); // commit
    const p = store.set(ensureClusterLabelAtom, 7);
    expect(api.pending.size).toBe(1);
    api.resolve(7, fakeLabel("X"));
    await p;
    expect(store.get(clusterLabelFor(7)).label?.label).toBe("X");
  });
});
