import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ClusterLabel } from "@/data";

import { SectionSpoken } from "./SectionSpoken";

afterEach(() => {
  cleanup();
});

const loaded = {
  detected_share: 0.42,
  top_langs: [
    { code: "en", share: 0.7 },
    { code: "es", share: 0.2 },
  ],
};

const audioLabel = (): ClusterLabel & { modality: "audio" } => ({
  label: "narration core",
  summary: "spoken-word delivery over minimal beds",
  modality: "audio",
  repertoire: [
    { tag: "direct address", description: "creators speak to camera", recurrence: "dominant" },
  ],
  aesthetic_logic: [
    { tag: "confessional intimacy", grounded_in: ["direct address"], description: "first-person framing" },
  ],
  taste_signalling: { label: "authenticity", description: "unscripted candour", confidence: "medium" },
  visibility_orientation: { label: "relatability", description: "low-production talking head", confidence: "low" },
  internal_variations: [],
  boundary_notes: "",
  tool_tags: ["voiceover", "talking-head"],
  validation: "ok",
  warnings: [],
});

describe("SectionSpoken", () => {
  it("renders language distribution without a label block", () => {
    render(<SectionSpoken index="03" loaded={loaded} />);
    expect(screen.getByText("EN")).toBeInTheDocument();
    expect(screen.queryByText("direct address")).not.toBeInTheDocument();
  });

  it("shows the empty state when no speech languages are detected", () => {
    render(<SectionSpoken index="03" loaded={{ detected_share: 0, top_langs: [] }} />);
    expect(screen.getByText("No language detected.")).toBeInTheDocument();
  });

  it("renders descriptive tags when an audio label block is present", () => {
    render(<SectionSpoken index="03" loaded={loaded} label={audioLabel()} />);
    expect(screen.getByText("direct address")).toBeInTheDocument();
    expect(screen.getByText("confessional intimacy")).toBeInTheDocument();
    // tool_tags ("voiceover") are no longer surfaced
    expect(screen.queryByText("voiceover")).not.toBeInTheDocument();
  });
});
