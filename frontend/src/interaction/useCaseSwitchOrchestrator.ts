import { useEffect } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import type { EmbeddingCase } from "@/data";
import {
  inspectorPhaseAtom,
  pendingCaseAtom,
  routeAtom,
  selectionAtom,
  type InspectorPhase,
  type Selection,
} from "@/state";

/**
 * Pure projection of the three watched atoms into an action.
 *
 *   - `noop`   — keep waiting; phase is still mid-close.
 *   - `apply`  — inspector reached `closed`; flip the route to `case`.
 *   - `abandon`— user re-opened a selection mid-close; drop the pending
 *     intent.
 *
 * Extracted from the hook so the rule is unit-testable without React.
 */
export type CaseSwitchOutcome =
  | { kind: "noop" }
  | { kind: "abandon" }
  | { kind: "apply"; case: EmbeddingCase };

export const resolveCaseSwitch = (
  pendingCase: EmbeddingCase | null,
  selection: Selection,
  phase: InspectorPhase,
): CaseSwitchOutcome => {
  if (pendingCase === null) return { kind: "noop" };
  if (selection !== null) return { kind: "abandon" };
  if (phase === "closed") return { kind: "apply", case: pendingCase };
  return { kind: "noop" };
};

/**
 * Defers a pending case switch until the inspector has finished its
 * close animation. Mounted once from `AppShell`.
 *
 * Coordinates four atoms:
 *
 *   - `pendingCaseAtom` — set by `VersionPill` when the user clicks a
 *     pill while a selection is open. Read-cleared here.
 *   - `selectionAtom` — `VersionPill` also sets this to `null` to
 *     start the close. Watched here so we abort the deferred switch
 *     if the user re-opens a selection mid-animation.
 *   - `inspectorPhaseAtom` — the trigger; the route flip only fires
 *     once the inspector reaches `closed`.
 *   - `routeAtom` — the eventual destination of the case write.
 *
 * Why a separate hook (and not extend `useVersionTransition`): the
 * version transition is driven by `transitionDriverAtom`, which is
 * itself seeded by `ensureRunAtom` once the route flips. The
 * orchestrator's job is to delay the route flip — i.e. it runs
 * upstream of every existing case-switch concern, and isolating it
 * keeps both pieces focused on one job.
 */
export const useCaseSwitchOrchestrator = (): void => {
  const pendingCase = useAtomValue(pendingCaseAtom);
  const selection = useAtomValue(selectionAtom);
  const phase = useAtomValue(inspectorPhaseAtom);
  const setRoute = useSetAtom(routeAtom);
  const setPendingCase = useSetAtom(pendingCaseAtom);

  useEffect(() => {
    const outcome = resolveCaseSwitch(pendingCase, selection, phase);
    if (outcome.kind === "apply") {
      setRoute((r) => ({ ...r, case: outcome.case }));
      setPendingCase(null);
    } else if (outcome.kind === "abandon") {
      setPendingCase(null);
    }
  }, [pendingCase, selection, phase, setRoute, setPendingCase]);
};
