import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ClusterLabel } from "@/data";

import { SectionTextual } from "./SectionTextual";

afterEach(() => {
  cleanup();
});

const loaded = {
  detected_share: 0.55,
  top_langs: [
    { code: "en", share: 0.6 },
    { code: "pt", share: 0.25 },
  ],
};

const textualLabel = (): ClusterLabel & { modality: "textual" } => ({
  label: "caption-driven",
  summary: "on-screen text carries the message",
  modality: "textual",
  repertoire: [
    { tag: "bold caption", description: "large overlaid headline text", recurrence: "dominant" },
  ],
  aesthetic_logic: [
    { tag: "punchline timing", grounded_in: ["bold caption"], description: "text reveals land the joke" },
  ],
  taste_signalling: { label: "wordplay", description: "verbal wit foregrounded", confidence: "medium" },
  visibility_orientation: { label: "shareability", description: "screenshot-friendly text", confidence: "low" },
  internal_variations: [],
  boundary_notes: "",
  tool_tags: ["text-overlay", "meme-format"],
  validation: "ok",
  warnings: [],
});

describe("SectionTextual", () => {
  it("renders language distribution without a label block", () => {
    render(<SectionTextual index="04" loaded={loaded} />);
    expect(screen.getByText("EN")).toBeInTheDocument();
    expect(screen.queryByText("bold caption")).not.toBeInTheDocument();
  });

  it("shows the empty state when no caption languages are present", () => {
    render(<SectionTextual index="04" loaded={{ detected_share: 0, top_langs: [] }} />);
    expect(screen.getByText("No captions in this slice.")).toBeInTheDocument();
  });

  it("renders descriptive tags when a textual label block is present", () => {
    render(<SectionTextual index="04" loaded={loaded} label={textualLabel()} />);
    expect(screen.getByText("bold caption")).toBeInTheDocument();
    expect(screen.getByText("punchline timing")).toBeInTheDocument();
    // tool_tags ("text-overlay") are no longer surfaced
    expect(screen.queryByText("text-overlay")).not.toBeInTheDocument();
  });
});
