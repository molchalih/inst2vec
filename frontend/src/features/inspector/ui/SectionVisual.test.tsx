import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ClusterLabel } from "@/data";

import { SectionVisual } from "./SectionVisual";

afterEach(() => {
  cleanup();
});

const baseLabel = (): ClusterLabel => ({
  label: "soft domestic",
  summary: "tight handheld kitchen scenes",
  modality: "visual",
  repertoire: [
    { tag: "warm kitchen", description: "tungsten domestic rooms", recurrence: "dominant" },
  ],
  aesthetic_logic: [
    { tag: "intimate realism", grounded_in: ["warm kitchen"], description: "intimate not staged" },
  ],
  taste_signalling: { label: "homecore", description: "low-stakes domestic affinity", confidence: "medium" },
  visibility_orientation: { label: "ordinariness", description: "no spectacle", confidence: "low" },
  internal_variations: [{ variation: "bathroom-lit", description: "minor strand of cool-lit grooming clips" }],
  boundary_notes: "differs from food-styling clusters",
  tool_tags: ["homecore", "warm-palette"],
  validation: "ok",
  warnings: [],
});

describe("SectionVisual cluster mode", () => {
  it("renders repertoire and aesthetic logic tags", () => {
    render(<SectionVisual index="05" cluster={baseLabel()} />);
    expect(screen.getByText("warm kitchen")).toBeInTheDocument();
    expect(screen.getByText("intimate realism")).toBeInTheDocument();
  });

  it("renders taste signalling and visibility orientation labels", () => {
    render(<SectionVisual index="05" cluster={baseLabel()} />);
    // "homecore" is the taste_signalling label
    expect(screen.getByText("homecore")).toBeInTheDocument();
    // "ordinariness" is the visibility_orientation label
    expect(screen.getByText("ordinariness")).toBeInTheDocument();
  });

  it("does not surface tool tags or a warning line", () => {
    const v = baseLabel();
    v.validation = "warn";
    v.warnings = ["invalid_confidence"];
    render(<SectionVisual index="05" cluster={v} />);
    expect(screen.queryByText("warm-palette")).not.toBeInTheDocument();
    expect(screen.queryByText(/invalid confidence/)).not.toBeInTheDocument();
    expect(screen.queryByText(/⚠/)).not.toBeInTheDocument();
  });

  it("caps a tag category at five chips", () => {
    const v = baseLabel();
    v.repertoire = Array.from({ length: 8 }, (_, i) => ({
      tag: `rep${i}`,
      description: "d",
      recurrence: "dominant" as const,
    }));
    render(<SectionVisual index="05" cluster={v} />);
    expect(screen.getByText("rep4")).toBeInTheDocument();
    expect(screen.queryByText("rep5")).not.toBeInTheDocument();
  });

  it("falls back to placeholder when neither clips nor cluster provided", () => {
    render(<SectionVisual index="05" />);
    expect(screen.getByText(/not yet available/i)).toBeInTheDocument();
  });

  it("renders 'Visual' title when modality is visual", () => {
    render(<SectionVisual index="05" cluster={baseLabel()} />);
    expect(screen.getByText("Visual")).toBeInTheDocument();
  });

  it("renders 'Audio' title when modality is audio", () => {
    render(<SectionVisual index="05" cluster={{ ...baseLabel(), modality: "audio" }} />);
    expect(screen.getByText("Audio")).toBeInTheDocument();
  });

  it("renders 'Musical' title when modality is music", () => {
    render(<SectionVisual index="05" cluster={{ ...baseLabel(), modality: "music" }} />);
    expect(screen.getByText("Musical")).toBeInTheDocument();
  });

  it("renders 'Combined' title when modality is multimodal", () => {
    render(<SectionVisual index="05" cluster={{ ...baseLabel(), modality: "multimodal" }} />);
    expect(screen.getByText("Combined")).toBeInTheDocument();
  });
});

describe("SectionVisual clip-only (creator) mode", () => {
  // Creator panes ship per-clip entries with no cluster. Those entries are
  // always the video clip-labels regardless of the active embedding case, so
  // the heading is fixed at "Visual" — never the run's modality.
  it("titles the per-clip list 'Visual'", () => {
    render(<SectionVisual index="05" clips={[]} />);
    expect(screen.getByText("Visual")).toBeInTheDocument();
  });
});
