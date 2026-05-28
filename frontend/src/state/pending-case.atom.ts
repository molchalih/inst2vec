import { atom } from "jotai";
import type { EmbeddingCase } from "@/data";
import { transitionAtom } from "./transition.atom";

/**
 * Queued case-switch intent. Written by `VersionPill` when the user
 * clicks a pill while the inspector is open; consumed by
 * `useCaseSwitchOrchestrator`, which applies the route change only
 * after the inspector has finished its close animation.
 *
 * Cleared by the orchestrator on either of two terminal conditions:
 *   - inspector phase reaches `closed` → the route flip is applied
 *     and the case-switch transition kicks off normally;
 *   - selection becomes non-null mid-close → the user has overridden
 *     their own pill click, so the pending intent is abandoned.
 */
export const pendingCaseAtom = atom<EmbeddingCase | null>(null);

/**
 * True from the moment the user clicks a pill (queued in
 * `pendingCaseAtom`) through the camera-flight transition's end. The
 * union covers the full switch lifecycle:
 *
 *   - **inspector-close window**: `pendingCaseAtom !== null` — the
 *     orchestrator is waiting for the inspector phase to reach
 *     `closed` before flipping the route.
 *   - **transition window**: `transitionAtom !== null` — the route
 *     has flipped, `ensureRun` has seeded the driver, and the
 *     camera-flight rAF loop is in progress.
 *
 * Consumed by `VersionPill` to disable inactive pills the instant a
 * click is registered, regardless of which window the switch is in.
 */
export const isCaseSwitchInFlightAtom = atom(
  (get) => get(pendingCaseAtom) !== null || get(transitionAtom) !== null,
);
