import { describe, expect, it } from "vitest";
import { batchItemSchema, nextBatchResponseSchema } from "./schemas";

describe("batchItemSchema", () => {
  it("parses a full batch item from the backend", () => {
    const parsed = nextBatchResponseSchema.parse({
      items: [
        {
          assignment_id: "asg-1",
          comparison_id: "c1",
          seed_group: "Artist",
          expected_modality: "caption_terms",
          creators: [
            { creator_id: 10, seed_group: "Artist", rep_clip_ids: [], caption_keywords: {}, audio_summary: {} },
            { creator_id: 20, seed_group: "Artist", rep_clip_ids: [1], caption_keywords: { keywords: ["paint"] }, audio_summary: {} },
            { creator_id: 30, seed_group: "Artist", rep_clip_ids: [], caption_keywords: {}, audio_summary: {} },
          ],
        },
      ],
    });
    expect(parsed.items[0]!.creators).toHaveLength(3);
  });

  it("rejects a comparison that is not exactly three creators", () => {
    expect(() =>
      batchItemSchema.parse({
        assignment_id: "a",
        comparison_id: "c",
        creators: [{ creator_id: 1 }, { creator_id: 2 }],
      }),
    ).toThrow();
  });
});
