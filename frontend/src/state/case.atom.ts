import { atom } from "jotai";
import type { EmbeddingCase } from "@/data";
import { pendingCaseAtom } from "./pending-case.atom";
import { routeAtom } from "./route.atom";

const DEFAULT_CASE: EmbeddingCase = "video";

/**
 * The actual active case derived from the URL route. Drives data
 * loading: `activeRunIdAtom` reads this to resolve the current run.
 */
export const caseAtom = atom<EmbeddingCase>(
  (get) => get(routeAtom).case ?? DEFAULT_CASE,
);

/**
 * The case to render as "selected" in the version drawer. Prefers the
 * queued case when one is pending so the pill highlight flips the
 * instant the user clicks, not when the deferred route change
 * eventually fires (~1.1s later, after the inspector close completes).
 *
 * Reverts to `caseAtom` automatically once the pending intent clears
 * — either when the orchestrator applies the route flip (then
 * `pendingCaseAtom` becomes null and `caseAtom` already equals the
 * previously-pending value) or when the user abandons the switch by
 * reopening a selection.
 */
export const displayedCaseAtom = atom<EmbeddingCase>(
  (get) => get(pendingCaseAtom) ?? get(caseAtom),
);
