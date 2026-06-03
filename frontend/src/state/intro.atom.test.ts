import { createStore } from "jotai";
import { describe, expect, it } from "vitest";
import {
  introAtom, introPlayedAtom, chromeRevealedAtom,
} from "./intro.atom";

describe("chromeRevealedAtom", () => {
  it("is hidden before the intro has played", () => {
    const store = createStore();
    expect(store.get(chromeRevealedAtom)).toBe(false);
  });

  it("stays hidden while the intro flight is in progress", () => {
    const store = createStore();
    store.set(introPlayedAtom, true);
    store.set(introAtom, { phase: 1, progress: 0.5, centerWorld: { x: 0, y: 0 } });
    expect(store.get(chromeRevealedAtom)).toBe(false);
  });

  it("reveals once the flight has settled (played, no frame in flight)", () => {
    const store = createStore();
    store.set(introPlayedAtom, true);
    store.set(introAtom, null);
    expect(store.get(chromeRevealedAtom)).toBe(true);
  });
});
