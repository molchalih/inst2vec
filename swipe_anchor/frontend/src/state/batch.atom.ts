import { atom, useAtomValue } from "jotai";
import type { BatchItem } from "@/data";
import { shouldRefill } from "@/core";
import { requireApiClient } from "./api-singleton";
import { crossedAtom } from "./crossed.atom";

const BATCH_N = 5;

export type BatchStatus = "idle" | "loading" | "error" | "exhausted";

/** Prefetched queue of comparisons; the head is the active one. */
export const batchQueueAtom = atom<BatchItem[]>([]);
export const batchStatusAtom = atom<BatchStatus>("idle");

/** Derived head of the queue (plan §8.3 currentItemAtom). */
export const currentItemAtom = atom<BatchItem | null>(
  (get) => get(batchQueueAtom)[0] ?? null,
);

/**
 * Refill the queue when it runs low (plan §8.3). Async action atom: atoms stay
 * synchronous to read; the fetch is the side effect, guarded against concurrent
 * refills via the status flag.
 */
export const ensureBatchAtom = atom(null, async (get, set) => {
  const queue = get(batchQueueAtom);
  if (!shouldRefill(queue.length)) return;
  if (get(batchStatusAtom) === "loading") return;

  set(batchStatusAtom, "loading");
  try {
    const res = await requireApiClient().getBatch(BATCH_N);
    const seen = new Set(get(batchQueueAtom).map((i) => i.assignment_id));
    const fresh = res.items.filter((i) => !seen.has(i.assignment_id));
    const merged = [...get(batchQueueAtom), ...fresh];
    set(batchQueueAtom, merged);
    set(batchStatusAtom, merged.length === 0 ? "exhausted" : "idle");
  } catch {
    set(batchStatusAtom, "error");
  }
});

/** Drop the active item and reset the crossed selection. */
export const advanceAtom = atom(null, (get, set) => {
  set(batchQueueAtom, get(batchQueueAtom).slice(1));
  set(crossedAtom, null);
});

export const useCurrentItem = () => useAtomValue(currentItemAtom);
export const useBatchStatus = () => useAtomValue(batchStatusAtom);
