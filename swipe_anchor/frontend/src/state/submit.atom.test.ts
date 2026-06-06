import { describe, expect, it } from "vitest";
import { createStore } from "jotai";
import type { ApiClient, BatchItem, RespondPayload } from "@/data";
import { setApiClient } from "./api-singleton";
import { batchQueueAtom } from "./batch.atom";
import { judgedCountAtom } from "./session.atom";
import { submitErrorAtom, submitJudgmentAtom } from "./submit.atom";

function makeItem(id: string): BatchItem {
  return {
    assignment_id: id,
    comparison_id: `cmp-${id}`,
    seed_group: "Artist",
    expected_modality: "caption_terms",
    creators: [
      { creator_id: 10, seed_group: "Artist", rep_clip_ids: [], caption_keywords: {}, audio_summary: {}, clips: [] },
      { creator_id: 20, seed_group: "Artist", rep_clip_ids: [], caption_keywords: {}, audio_summary: {}, clips: [] },
      { creator_id: 30, seed_group: "Artist", rep_clip_ids: [], caption_keywords: {}, audio_summary: {}, clips: [] },
    ],
  };
}

const input = {
  oddId: 30,
  confidence: 1,
  reactionMs: 500,
  dwell: null,
  expanded: false,
} as const;

function fakeClient(over: Partial<ApiClient>): ApiClient {
  return {
    getBatch: async () => ({ items: [] }),
    respond: async (_p: RespondPayload) => ({ accepted: true, n_triplets: 2, retired: false }),
    ...over,
  };
}

describe("submitJudgmentAtom", () => {
  it("advances and counts on a successful submit", async () => {
    setApiClient(fakeClient({}));
    const store = createStore();
    store.set(batchQueueAtom, [makeItem("a"), makeItem("b")]);

    await store.set(submitJudgmentAtom, input);

    expect(store.get(batchQueueAtom).map((i) => i.assignment_id)).toEqual(["b"]);
    expect(store.get(judgedCountAtom)).toBe(1);
    expect(store.get(submitErrorAtom)).toBe(false);
  });

  it("rolls back the optimistic advance when the submit fails", async () => {
    setApiClient(
      fakeClient({
        respond: async () => {
          throw new Error("offline");
        },
      }),
    );
    const store = createStore();
    store.set(batchQueueAtom, [makeItem("a"), makeItem("b")]);

    await expect(store.set(submitJudgmentAtom, input)).rejects.toThrow("offline");

    // The failed comparison is back at the head; the count is undone; error flagged.
    expect(store.get(batchQueueAtom)[0]!.assignment_id).toBe("a");
    expect(store.get(judgedCountAtom)).toBe(0);
    expect(store.get(submitErrorAtom)).toBe(true);
  });
});
