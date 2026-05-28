import { describe, it, expect } from "vitest";
import type { Selection, InspectorPhase } from "@/state";
import { resolveCaseSwitch } from "./useCaseSwitchOrchestrator";

const noSel: Selection = null;
const someSel: Selection = { kind: "cluster", clusterId: 7 };

describe("resolveCaseSwitch", () => {
  it("no-op when nothing is pending", () => {
    expect(resolveCaseSwitch(null, someSel, "open")).toEqual({ kind: "noop" });
    expect(resolveCaseSwitch(null, noSel, "closed")).toEqual({ kind: "noop" });
  });

  it("holds while phase is any non-closed value", () => {
    const phases: InspectorPhase[] = [
      "opening-slide", "opening-content", "open",
      "closing-content", "closing-slide",
    ];
    for (const p of phases) {
      expect(resolveCaseSwitch("sandwich", noSel, p)).toEqual({ kind: "noop" });
    }
  });

  it("applies when selection cleared and phase reached `closed`", () => {
    expect(resolveCaseSwitch("sandwich", noSel, "closed"))
      .toEqual({ kind: "apply", case: "sandwich" });
  });

  it("abandons when the user re-opened a selection mid-close", () => {
    expect(resolveCaseSwitch("sandwich", someSel, "closing-content"))
      .toEqual({ kind: "abandon" });
    expect(resolveCaseSwitch("sandwich", someSel, "open"))
      .toEqual({ kind: "abandon" });
  });
});
