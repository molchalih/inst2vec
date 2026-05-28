import { describe, it, expect } from "vitest";
import { clipLabelEntrySchema } from "./clip-label.schema";

const minimal = {
  clip_id: 1,
  shortcode: "ABC",
  thumbnail_url: "https://x/y.jpg",
  sentence: "warm kitchen scene with shallow depth of field",
  tags: {
    observable: [{ tag: "warm kitchen", evidence: "lamp" }],
    aesthetic: [
      { tag: "soft vignette", grounded_in: ["warm kitchen"], confidence: "medium" as const },
    ],
    community: [
      { tag: "homecore", grounded_in: ["soft vignette"], confidence: "low" as const },
    ],
  },
  validation: "ok" as const,
  warnings: [],
};

describe("clipLabelEntrySchema", () => {
  it("accepts a clean payload", () => {
    expect(() => clipLabelEntrySchema.parse(minimal)).not.toThrow();
  });

  it("accepts validation=warn with warnings list", () => {
    const warn = { ...minimal, validation: "warn" as const, warnings: ["tag_count_out_of_range"] };
    expect(() => clipLabelEntrySchema.parse(warn)).not.toThrow();
  });

  it("accepts validation=warn payload with non-enum confidence (S6)", () => {
    const warn = {
      ...minimal,
      validation: "warn" as const,
      warnings: ["S6"],
      tags: {
        ...minimal.tags,
        aesthetic: [
          { tag: "soft vignette", grounded_in: ["warm kitchen"], confidence: "definitely" },
        ],
      },
    };
    expect(() => clipLabelEntrySchema.parse(warn)).not.toThrow();
  });

  it("rejects unknown validation enum", () => {
    expect(() =>
      clipLabelEntrySchema.parse({ ...minimal, validation: "bogus" as never }),
    ).toThrow();
  });

  it("rejects observable tag missing evidence", () => {
    const bad = {
      ...minimal,
      tags: { ...minimal.tags, observable: [{ tag: "warm kitchen" }] as never },
    };
    expect(() => clipLabelEntrySchema.parse(bad)).toThrow();
  });
});
