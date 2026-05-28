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
    // "homecore" appears in taste_signalling.label and tool_tags — use getAllByText
    expect(screen.getAllByText("homecore").length).toBeGreaterThan(0);
    // "ordinariness" is the visibility_orientation label
    expect(screen.getByText("ordinariness")).toBeInTheDocument();
  });

  it("renders tool_tags as a passive row", () => {
    render(<SectionVisual index="05" cluster={baseLabel()} />);
    // "homecore" also appears as taste_signalling label, so use getAllByText
    expect(screen.getAllByText("homecore").length).toBeGreaterThan(0);
    expect(screen.getByText("warm-palette")).toBeInTheDocument();
  });

  it("renders warning line in warn mode with translated codes", () => {
    const v = baseLabel();
    v.validation = "warn";
    v.warnings = ["invalid_confidence"];
    render(<SectionVisual index="05" cluster={v} />);
    expect(screen.getByText(/invalid confidence/)).toBeInTheDocument();
  });

  it("falls back to TODO placeholder when neither clips nor cluster provided", () => {
    render(<SectionVisual index="05" />);
    expect(screen.getByText(/TODO/)).toBeInTheDocument();
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

describe("SectionVisual modality fallback (clip-only)", () => {
  // Creator pane ships per-clip entries with no cluster, so the title used to
  // fall through to "Visual" for every run. Passing modality on the props
  // alongside `clips` keeps the heading honest for audio / music / multimodal.
  it("uses the modality prop when no cluster is supplied", () => {
    render(<SectionVisual index="05" clips={[]} modality="audio" />);
    expect(screen.getByText("Audio")).toBeInTheDocument();
  });

  it("modality prop also titles the empty placeholder branch", () => {
    render(<SectionVisual index="05" modality="music" />);
    expect(screen.getByText("Musical")).toBeInTheDocument();
  });

  it("cluster.modality wins over the modality prop", () => {
    render(
      <SectionVisual
        index="05"
        cluster={{ ...baseLabel(), modality: "multimodal" }}
        modality="audio"
      />,
    );
    expect(screen.getByText("Combined")).toBeInTheDocument();
  });
});
