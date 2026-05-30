import { describe, expect, it } from "vitest";
import { createStore } from "jotai";
import { viewportAtom, focusViewportAtom } from "./viewport.atom";
import { runStateAtom } from "./run.atom";
import { viewportSizeAtom } from "./viewport-size.atom";
import type { AtlasRun } from "@/data";

const run: AtlasRun = {
  meta: { id: "video-1", case: "video", label: "v", size: 1, details_available: false },
  bounds: { minX: -1, maxX: 1, minY: -1, maxY: 1 },
  users: [[1, 0, 0, 0, false, 0.5]],
  clusters: [],
};

const seed = (store: ReturnType<typeof createStore>) => {
  store.set(runStateAtom, { runs: new Map([["video-1", run]]), activeRunId: "video-1" });
  store.set(viewportSizeAtom, { width: 1000, height: 1000 });
};

// The camera-focus tween writes interpolated frames here. It must pass
// them through verbatim: the tween interpolates between two pre-validated
// endpoints, so re-clamping each frame against the current fit band is
// wrong. Regression for the deselect snap — a large cluster focuses below
// the full-viewport fit floor, and clamping each frame up to that floor
// pinned the scale instead of letting it ease back to the fit.
describe("focusViewportAtom — camera tween write path", () => {
  it("passes a scale below the fit-band floor through unclamped", () => {
    const store = createStore();
    seed(store);
    const fit = store.get(viewportAtom).scale; // = the band floor (minScaleFactor = 1)
    const below = fit * 0.5;
    store.set(focusViewportAtom, { x: 0, y: 0, scale: below });
    expect(store.get(viewportAtom).scale).toBeCloseTo(below);
  });

  it("passes pan through unclamped", () => {
    const store = createStore();
    seed(store);
    const fit = store.get(viewportAtom).scale;
    store.set(focusViewportAtom, { x: 99_999, y: -99_999, scale: fit });
    expect(store.get(viewportAtom).x).toBeCloseTo(99_999);
    expect(store.get(viewportAtom).y).toBeCloseTo(-99_999);
  });
});
