import { atom } from "jotai";
import type { BatchItem, RespondResponse } from "@/data";
import { requireApiClient } from "./api-singleton";
import {
  advanceAtom,
  batchQueueAtom,
  currentItemAtom,
  ensureBatchAtom,
} from "./batch.atom";
import { judgedCountAtom } from "./session.atom";

export interface JudgmentInput {
  /** crossed creator id, or null for "skip / too close to call" */
  oddId: number | null;
  confidence: number;
  reactionMs: number | null;
  dwell: Record<number, number> | null;
  expanded: boolean;
}

/** Set when the most recent submission failed to reach the backend. */
export const submitErrorAtom = atom<boolean>(false);

/**
 * Submit a judgment for the active comparison (plan §8.3). Advances optimistically
 * so the next card is instant, then POSTs. If the POST fails (offline / 5xx) the
 * advance is **rolled back** — the item is restored to the head of the queue, the
 * session count is decremented, and `submitErrorAtom` is set — so a transient
 * failure never silently drops a judgment. (Server-side idempotency makes a later
 * re-submit safe even if the original actually landed.)
 */
export const submitJudgmentAtom = atom(
  null,
  async (get, set, input: JudgmentInput): Promise<RespondResponse | null> => {
    const item: BatchItem | null = get(currentItemAtom);
    if (!item) return null;

    set(submitErrorAtom, false);
    set(advanceAtom); // optimistic advance (also resets the crossed selection)
    set(judgedCountAtom, get(judgedCountAtom) + 1);
    void set(ensureBatchAtom);

    // The reel shown for each creator is the first clip on its card (clips[0]);
    // capture it so each judgment records exactly which reels were compared.
    const shownClips: Record<number, number> = {};
    for (const c of item.creators) {
      const clipId = c.clips[0]?.clip_id;
      if (clipId != null) shownClips[c.creator_id] = clipId;
    }

    try {
      return await requireApiClient().respond({
        assignment_id: item.assignment_id,
        odd_creator_id: input.oddId,
        confidence: input.confidence,
        reaction_time_ms: input.reactionMs,
        card_dwell_ms: input.dwell,
        shown_clips: Object.keys(shownClips).length ? shownClips : null,
        expanded: input.expanded,
      });
    } catch (err) {
      // Restore the failed comparison to the front and undo the count bump.
      const queue = get(batchQueueAtom);
      if (!queue.some((i) => i.assignment_id === item.assignment_id)) {
        set(batchQueueAtom, [item, ...queue]);
      }
      set(judgedCountAtom, Math.max(0, get(judgedCountAtom) - 1));
      set(submitErrorAtom, true);
      throw err;
    }
  },
);
