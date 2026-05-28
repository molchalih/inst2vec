import { describe, it, expect } from "vitest";
import {
  type InspectorPhase,
  isContentMounted,
  isPanelOpen,
  shouldAdvanceDisplayed,
} from "./inspector-phase.atom";

const ALL: InspectorPhase[] = [
  "closed",
  "opening-slide",
  "opening-content",
  "open",
  "closing-content",
  "closing-slide",
];

describe("inspector phase helpers", () => {
  it("isPanelOpen: panel mounted during slide-in, content phases, and steady", () => {
    const byPhase = Object.fromEntries(ALL.map((p) => [p, isPanelOpen(p)]));
    expect(byPhase).toEqual({
      "closed": false,
      "opening-slide": true,
      "opening-content": true,
      "open": true,
      "closing-content": true,
      "closing-slide": false,
    });
  });

  it("shouldAdvanceDisplayed: true only during entering sub-phases — excludes steady `open` so a fresh click can't leak into the previous pane before the close animation begins", () => {
    const advanced = ALL.filter(shouldAdvanceDisplayed);
    expect(advanced).toEqual(["opening-slide", "opening-content"]);
  });

  it("isContentMounted: includes closing-slide so the --out animation overlaps the panel slide-out", () => {
    const mounted = ALL.filter(isContentMounted);
    expect(mounted).toEqual([
      "opening-content",
      "open",
      "closing-content",
      "closing-slide",
    ]);
  });
});
