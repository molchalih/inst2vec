import { describe, expect, it } from "vitest";
import { createStore } from "jotai";
import { trackedPresenceAtom } from "./tracked-presence.atom";
import { trackedCreatorAtom } from "./tracking.atom";
import { manifestAtom } from "./manifest.atom";
import { runStateAtom } from "./run.atom";
import type { AtlasRun, Manifest } from "@/data";

const manifest = (runIds: [string, ...string[]]): Manifest => ({
  version: 6,
  default_run_id: runIds[0],
  runs: runIds.map((id) => ({
    id, case: "video", label: id, size: 1, details_available: true,
  })),
});

const runWith = (id: string, creatorIds: number[]): AtlasRun => ({
  meta: { id, case: "video", label: id, size: creatorIds.length, details_available: false },
  bounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 },
  users: creatorIds.map((cid) => [cid, 0, 0, -1, false, 0]),
  clusters: [],
});

describe("tracked-presence.atom", () => {
  it("reports every run present when nothing is tracked", () => {
    const store = createStore();
    store.set(manifestAtom, manifest(["run-1", "run-2"]));
    expect(store.get(trackedPresenceAtom)).toEqual({ "run-1": true, "run-2": true });
  });

  it("treats an uncached run as absent while a creator is tracked", () => {
    const store = createStore();
    store.set(manifestAtom, manifest(["run-1"]));
    store.set(trackedCreatorAtom, 42);
    // runStateAtom cache is empty → unknown → pessimistic false (hard invariant).
    expect(store.get(trackedPresenceAtom)).toEqual({ "run-1": false });
  });

  it("reports present for a cached run containing the tracked creator", () => {
    const store = createStore();
    store.set(manifestAtom, manifest(["run-1"]));
    store.set(trackedCreatorAtom, 42);
    store.set(runStateAtom, {
      runs: new Map([["run-1", runWith("run-1", [42])]]),
      activeRunId: "run-1",
    });
    expect(store.get(trackedPresenceAtom)).toEqual({ "run-1": true });
  });

  it("reports absent for a cached run lacking the tracked creator", () => {
    const store = createStore();
    store.set(manifestAtom, manifest(["run-1"]));
    store.set(trackedCreatorAtom, 42);
    store.set(runStateAtom, {
      runs: new Map([["run-1", runWith("run-1", [7, 99])]]),
      activeRunId: "run-1",
    });
    expect(store.get(trackedPresenceAtom)).toEqual({ "run-1": false });
  });
});
