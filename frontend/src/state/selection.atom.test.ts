import { describe, expect, it } from "vitest";
import { createStore } from "jotai";
import {
  selectionAtom, selectDotAtom, selectClusterAtom,
} from "./selection.atom";
import { runStateAtom } from "./run.atom";
import { manifestAtom } from "./manifest.atom";
import { viewportAtom } from "./viewport.atom";
import { viewportSizeAtom } from "./viewport-size.atom";
import type { AtlasRun, Manifest } from "@/data";

const runWith = (hasDetail: boolean): AtlasRun => ({
  meta: { id: "video-1", case: "video", label: "v", size: 2, details_available: hasDetail },
  bounds: { minX: -1, maxX: 1, minY: -1, maxY: 1 },
  users: [
    [1, 0, 0, 5, hasDetail, 0.7],
    [2, 1, 1, -1, false, 0],
    [3, 0.5, 0.5, 5, false, 0.5],  // dot in cluster, but its row has no detail
  ],
  clusters: [],
});

const manifestWith = (details: boolean): Manifest => ({
  version: 3,
  default_run_id: "video-1",
  runs: [{ id: "video-1", case: "video", label: "v", size: 2, details_available: details }],
});

const seed = (store: ReturnType<typeof createStore>, details: boolean) => {
  store.set(runStateAtom, { runs: new Map([["video-1", runWith(details)]]), activeRunId: "video-1" });
  store.set(manifestAtom, manifestWith(details));
};

describe("selection.atom — selectDotAtom", () => {
  // TODO: restore creator selection once per-creator data is ready
  // (the creator path in selectDotAtom is commented out for now).
  it("details on + has_detail true → cluster selection (creator path disabled)", () => {
    const store = createStore();
    seed(store, true);
    store.set(selectDotAtom, 1);
    expect(store.get(selectionAtom)).toEqual({ kind: "cluster", clusterId: 5 });
  });

  it("details on + has_detail false → cluster fallback", () => {
    const store = createStore();
    seed(store, true);
    store.set(selectDotAtom, 3);
    expect(store.get(selectionAtom)).toEqual({ kind: "cluster", clusterId: 5 });
  });

  it("details off → cluster fallback regardless of has_detail", () => {
    const store = createStore();
    seed(store, false);
    store.set(selectDotAtom, 1);
    expect(store.get(selectionAtom)).toEqual({ kind: "cluster", clusterId: 5 });
  });

  it("noise dot → no-op", () => {
    const store = createStore();
    seed(store, true);
    store.set(selectDotAtom, 2);
    expect(store.get(selectionAtom)).toBeNull();
  });

  it("unknown dot id → no-op", () => {
    const store = createStore();
    seed(store, true);
    store.set(selectDotAtom, 999);
    expect(store.get(selectionAtom)).toBeNull();
  });

  it("no active run → no-op", () => {
    const store = createStore();
    store.set(selectDotAtom, 1);
    expect(store.get(selectionAtom)).toBeNull();
  });

  it("pins viewport on select", () => {
    const store = createStore();
    seed(store, true);
    store.set(viewportSizeAtom, { width: 1000, height: 1000 });
    const before = store.get(viewportAtom);
    store.set(selectDotAtom, 1);
    expect(store.get(viewportAtom)).toEqual(before);
  });
});

describe("selection.atom — selectClusterAtom", () => {
  it("writes a cluster selection and pins the viewport", () => {
    const store = createStore();
    seed(store, true);
    store.set(viewportSizeAtom, { width: 1000, height: 1000 });
    const before = store.get(viewportAtom);
    store.set(selectClusterAtom, 9);
    expect(store.get(selectionAtom)).toEqual({ kind: "cluster", clusterId: 9 });
    expect(store.get(viewportAtom)).toEqual(before);
  });
});
