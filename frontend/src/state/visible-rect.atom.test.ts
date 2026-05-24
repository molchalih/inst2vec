import { describe, expect, it } from "vitest";
import { createStore } from "jotai";
import { visibleRectAtom } from "./visible-rect.atom";
import { selectionAtom } from "./selection.atom";
import { viewportSizeAtom } from "./viewport-size.atom";
import { tokens } from "@/ui/tokens";

describe("visibleRectAtom", () => {
  it("equals the viewport when selection is null", () => {
    const store = createStore();
    store.set(viewportSizeAtom, { width: 1200, height: 800 });
    expect(store.get(visibleRectAtom)).toEqual({
      x: 0, y: 0, width: 1200, height: 800,
    });
  });

  it("insets by panel width when selection is non-null", () => {
    const store = createStore();
    store.set(viewportSizeAtom, { width: 1200, height: 800 });
    store.set(selectionAtom, { kind: "cluster", clusterId: 0 });
    const w = tokens.panel.widthPx;
    expect(store.get(visibleRectAtom)).toEqual({
      x: w, y: 0, width: 1200 - w, height: 800,
    });
  });

  it("clamps width to zero when the viewport is narrower than the panel", () => {
    const store = createStore();
    store.set(viewportSizeAtom, { width: 100, height: 800 });
    store.set(selectionAtom, { kind: "cluster", clusterId: 0 });
    expect(store.get(visibleRectAtom).width).toBe(0);
  });
});
