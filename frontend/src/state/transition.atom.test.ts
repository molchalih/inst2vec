import { describe, expect, it } from "vitest";
import { createStore } from "jotai";
import type { AtlasRun } from "@/data";
import { isTransitioningAtom, transitionAtom } from "./transition.atom";

const fakeRun = (id: string): AtlasRun => ({
  meta: { id, case: "video", label: id, size: 0, details_available: false },
  bounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 },
  users: [],
  clusters: [],
});

describe("transitionAtom", () => {
  it("defaults to null", () => {
    const store = createStore();
    expect(store.get(transitionAtom)).toBeNull();
  });

  it("accepts a transition state and round-trips it", () => {
    const store = createStore();
    const t = { from: fakeRun("a"), to: fakeRun("b"), phase: 1 as const, progress: 0.5 };
    store.set(transitionAtom, t);
    expect(store.get(transitionAtom)).toEqual(t);
  });

  it("clears back to null", () => {
    const store = createStore();
    store.set(transitionAtom, { from: fakeRun("a"), to: fakeRun("b"), phase: 0 as const, progress: 0 });
    store.set(transitionAtom, null);
    expect(store.get(transitionAtom)).toBeNull();
  });
});

describe("isTransitioningAtom", () => {
  it("is false when transitionAtom is null", () => {
    const store = createStore();
    expect(store.get(isTransitioningAtom)).toBe(false);
  });

  it("flips to true while transitionAtom carries a state", () => {
    const store = createStore();
    store.set(transitionAtom, {
      from: fakeRun("a"), to: fakeRun("b"),
      phase: 1 as const, progress: 0.3,
    });
    expect(store.get(isTransitioningAtom)).toBe(true);
    store.set(transitionAtom, null);
    expect(store.get(isTransitioningAtom)).toBe(false);
  });
});
