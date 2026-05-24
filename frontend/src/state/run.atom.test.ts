import { describe, expect, it, beforeEach } from "vitest";
import { createStore } from "jotai";
import type { AtlasRun, BulkSource, Manifest } from "@/data";
import { setBulkSource } from "./bulk-singleton";
import { ensureRunAtom, runStateAtom } from "./run.atom";
import { transitionAtom, transitionDriverAtom } from "./transition.atom";
import { viewportSizeAtom } from "./viewport-size.atom";

type Deferred<T> = { promise: Promise<T>; resolve: (v: T) => void };

const defer = <T>(): Deferred<T> => {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((res) => { resolve = res; });
  return { promise, resolve };
};

const fakeRun = (id: string): AtlasRun => ({
  meta: { id, case: "video", label: id, size: 0, details_available: false },
  bounds: { minX: 0, minY: 0, maxX: 1, maxY: 1 },
  users: [],
  clusters: [],
});

class ProgrammableBulkSource implements BulkSource {
  pending = new Map<string, Deferred<AtlasRun>>();

  async getManifest(): Promise<Manifest> {
    throw new Error("not used");
  }

  getRun(runId: string): Promise<AtlasRun> {
    const d = defer<AtlasRun>();
    this.pending.set(runId, d);
    return d.promise;
  }

  resolve(runId: string): void {
    const d = this.pending.get(runId);
    if (!d) throw new Error(`no pending fetch for ${runId}`);
    d.resolve(fakeRun(runId));
    this.pending.delete(runId);
  }
}

describe("ensureRunAtom", () => {
  let bulk: ProgrammableBulkSource;

  beforeEach(() => {
    bulk = new ProgrammableBulkSource();
    setBulkSource(bulk);
  });

  it("caches the loaded run and activates it", async () => {
    const store = createStore();
    const p = store.set(ensureRunAtom, "video-1");
    bulk.resolve("video-1");
    await p;
    const state = store.get(runStateAtom);
    expect(state.activeRunId).toBe("video-1");
    expect(state.runs.get("video-1")?.meta.id).toBe("video-1");
  });

  it("the latest request wins when an earlier fetch resolves after a newer one", async () => {
    const store = createStore();
    const slow = store.set(ensureRunAtom, "video-1");
    const fast = store.set(ensureRunAtom, "audio-1");

    // Resolve in reverse order: the newer request finishes first.
    bulk.resolve("audio-1");
    await fast;
    bulk.resolve("video-1");
    await slow;

    const state = store.get(runStateAtom);
    expect(state.activeRunId).toBe("audio-1");
    // Both runs landed in the cache; the slow load did not drop the newer entry.
    expect(state.runs.has("audio-1")).toBe(true);
    expect(state.runs.has("video-1")).toBe(true);
  });

  it("does not refetch a run that is already cached", async () => {
    const store = createStore();
    const p = store.set(ensureRunAtom, "video-1");
    bulk.resolve("video-1");
    await p;

    // Second call: the bulk source must not be hit at all.
    await store.set(ensureRunAtom, "video-1");
    expect(bulk.pending.size).toBe(0);
    expect(store.get(runStateAtom).activeRunId).toBe("video-1");
  });
});

describe("ensureRunAtom (transition seeding)", () => {
  let bulk: ProgrammableBulkSource;

  beforeEach(() => {
    bulk = new ProgrammableBulkSource();
    setBulkSource(bulk);
  });

  it("seeds transitionAtom and transitionDriverAtom in the same tick as the activeRunId flip", async () => {
    const store = createStore();
    store.set(viewportSizeAtom, { width: 1000, height: 800 });

    const first = store.set(ensureRunAtom, "video-1");
    bulk.resolve("video-1");
    await first;
    expect(store.get(transitionAtom)).toBeNull();
    expect(store.get(transitionDriverAtom)).toBeNull();

    const second = store.set(ensureRunAtom, "audio-1");
    bulk.resolve("audio-1");
    await second;

    const t = store.get(transitionAtom);
    const d = store.get(transitionDriverAtom);
    expect(t).not.toBeNull();
    expect(t!.from.meta.id).toBe("video-1");
    expect(t!.to.meta.id).toBe("audio-1");
    expect(t!.phase).toBe(0);
    expect(t!.progress).toBe(0);

    expect(d).not.toBeNull();
    expect(d!.from.meta.id).toBe("video-1");
    expect(d!.to.meta.id).toBe("audio-1");
    expect(typeof d!.startTime).toBe("number");
  });

  it("blocks further switches while a transition is in flight", async () => {
    const store = createStore();
    store.set(viewportSizeAtom, { width: 1000, height: 800 });

    const first = store.set(ensureRunAtom, "video-1");
    bulk.resolve("video-1");
    await first;
    const second = store.set(ensureRunAtom, "audio-1");
    bulk.resolve("audio-1");
    await second;

    // Now a transition is mid-flight: from=video-1, to=audio-1.
    expect(store.get(transitionAtom)).not.toBeNull();
    const beforeId = store.get(runStateAtom).activeRunId;

    // A third request must be ignored entirely.
    await store.set(ensureRunAtom, "sandwich-1");
    expect(store.get(runStateAtom).activeRunId).toBe(beforeId);
    // Transition state unchanged.
    expect(store.get(transitionAtom)!.to.meta.id).toBe("audio-1");
  });

  it("does not seed a transition when activating the same run id twice", async () => {
    const store = createStore();
    store.set(viewportSizeAtom, { width: 1000, height: 800 });

    const p1 = store.set(ensureRunAtom, "video-1");
    bulk.resolve("video-1");
    await p1;

    store.set(transitionAtom, null);
    store.set(transitionDriverAtom, null);

    await store.set(ensureRunAtom, "video-1");
    expect(store.get(transitionAtom)).toBeNull();
    expect(store.get(transitionDriverAtom)).toBeNull();
  });
});
