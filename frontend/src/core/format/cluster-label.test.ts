import { describe, expect, it } from "vitest";
import { clusterSummaryLede, clusterWarningLabel } from "./cluster-label";

describe("clusterWarningLabel", () => {
  it("translates known codes", () => {
    expect(clusterWarningLabel("tag_count_out_of_range")).toBe("tag count out of range");
    expect(clusterWarningLabel("invalid_confidence")).toBe("invalid confidence");
    expect(clusterWarningLabel("ungrounded_tag_reference")).toBe("ungrounded tag reference");
    expect(clusterWarningLabel("invalid_tool_tags")).toBe("invalid tool tags");
  });

  it("passes unknown codes through", () => {
    expect(clusterWarningLabel("ZZ9")).toBe("ZZ9");
  });
});

describe("clusterSummaryLede", () => {
  it("strips the boilerplate lead-in and re-capitalises", () => {
    expect(
      clusterSummaryLede("The shared musical identity combines electronic and melodic elements."),
    ).toBe("Electronic and melodic elements.");
    expect(
      clusterSummaryLede("The shared spoken identity centers on personal storytelling."),
    ).toBe("Personal storytelling.");
    expect(
      clusterSummaryLede("The shared visual identity revolves around neon nightlife shots."),
    ).toBe("Neon nightlife shots.");
  });

  it("handles connector and spelling variants", () => {
    expect(clusterSummaryLede("A shared musical identity centred on ambient soundscapes.")).toBe(
      "Ambient soundscapes.",
    );
    expect(clusterSummaryLede("The shared musical identity features a focus on rhythm.")).toBe(
      "A focus on rhythm.",
    );
  });

  it("passes summaries without the skeleton through untouched", () => {
    const s = "A blend of cinematic and electronic elements with a focus on mood.";
    expect(clusterSummaryLede(s)).toBe(s);
  });
});
