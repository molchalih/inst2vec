import { useAtomValue, useSetAtom } from "jotai";
import {
  caseAtom, displayedCaseAtom, isCaseSwitchInFlightAtom, pendingCaseAtom,
  routeAtom, selectionAtom, useIsIntroPlaying, useTrackedCreator,
  useTrackedPresentInRun, type Route,
} from "@/state";
import type { ManifestRun } from "@/data";

type Props = { run: ManifestRun };

/**
 * Single embedding-case pill. Clicking sets routeAtom.case directly when
 * no inspector selection is open; otherwise it queues a deferred switch
 * via `pendingCaseAtom` + clears `selectionAtom` so the inspector close
 * animation runs first. `useCaseSwitchOrchestrator` (mounted in
 * AppShell) applies the queued route change once the inspector phase
 * reaches `closed`. The existing caseAtom → activeRunIdAtom →
 * RunLoader chain then handles fetch + activate as before.
 *
 * Active pill is filled; inactive is outlined. Disabled while a
 * version-switch transition is in flight (active pill stays live so
 * the row never looks fully dead) and while the entrance flight
 * plays — UX half of the guard; ensureRunAtom enforces the switch
 * rule at the writer.
 */
export const VersionPill = ({ run }: Props) => {
  // Reads `displayedCaseAtom` (not `caseAtom`) so the pill highlight
  // tracks the user's clicked intent immediately, not the deferred
  // route flip.
  const displayedCase = useAtomValue(displayedCaseAtom);
  const setRoute = useSetAtom(routeAtom);
  const setPendingCase = useSetAtom(pendingCaseAtom);
  const setSelection = useSetAtom(selectionAtom);
  const selection = useAtomValue(selectionAtom);
  // Single source of truth across both windows of the switch lifecycle
  // (inspector close + camera flight). Flips to true the moment the
  // user clicks any pill so the row freezes immediately, not when the
  // route eventually changes.
  const isSwitching = useAtomValue(isCaseSwitchInFlightAtom);
  const introPlaying = useIsIntroPlaying();
  const isActive = run.case === displayedCase;
  // Point-tracking presence gate: a case lacking the tracked creator is
  // unreachable. Never disables the *committed* active pill (you're already on
  // a present case — tracking can only start on a dot in the active run). The
  // exemption keys on `caseAtom` (the committed route case), NOT
  // `displayedCaseAtom`, since the latter includes a queued `pendingCase` whose
  // target must still be presence-gated until the switch actually commits.
  const committedCase = useAtomValue(caseAtom);
  const trackedCreatorId = useTrackedCreator();
  const presentHere = useTrackedPresentInRun(run.id);
  const trackingDisablesThisCase =
    trackedCreatorId != null && run.case !== committedCase && !presentHere;
  const disabled = (isSwitching && !isActive) || introPlaying || trackingDisablesThisCase;
  const onClick = (): void => {
    if (isActive) return;
    if (selection === null) {
      setRoute((prev: Route) => ({ ...prev, case: run.case }));
      return;
    }
    // Defer: trigger the inspector close, queue the case switch for
    // when phase=closed (see useCaseSwitchOrchestrator).
    setPendingCase(run.case);
    setSelection(null);
  };
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={isActive}
      disabled={disabled}
      className={[
        "pointer-events-auto",
        "rounded-pill px-pill-px py-pill-py text-sm",
        "backdrop-blur-glass border border-fg-default/10 shadow-glass",
        // Transition both colours AND opacity so the disabled fade
        // animates instead of snapping. `transition` (vs.
        // `transition-colors`) covers the same standard properties
        // plus opacity. `duration-medium` gives the morph enough
        // weight to feel intentional without dragging the click
        // affordance.
        "transition duration-medium ease-motion-out",
        "disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-bg-canvas/35",
        isActive
          ? "bg-fg-default/85 text-bg-canvas"
          : "bg-bg-canvas/35 text-fg-default hover:bg-bg-canvas/50",
      ].join(" ")}
    >
      {run.label}
    </button>
  );
};
