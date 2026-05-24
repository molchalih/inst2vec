import { useAtomValue, useSetAtom } from "jotai";
import {
  caseAtom, routeAtom, useIsTransitioning, useIsIntroPlaying, type Route,
} from "@/state";
import type { ManifestRun } from "@/data";

type Props = { run: ManifestRun };

/**
 * Single embedding-case pill. Clicking sets routeAtom.case; the existing
 * caseAtom → activeRunIdAtom → RunLoader chain handles fetch + activate.
 * Active pill is filled; inactive is outlined. The pill is disabled
 * while a version-switch transition is in flight (active pill stays live
 * so the row never looks fully dead) and while the entrance flight plays
 * (all pills locked until the dots settle) — the UX half of the guard;
 * ensureRunAtom enforces the switch rule at the writer.
 */
export const VersionPill = ({ run }: Props) => {
  const activeCase = useAtomValue(caseAtom);
  const setRoute = useSetAtom(routeAtom);
  const isTransitioning = useIsTransitioning();
  const introPlaying = useIsIntroPlaying();
  const isActive = run.case === activeCase;
  const disabled = (isTransitioning && !isActive) || introPlaying;
  const onClick = (): void => {
    setRoute((prev: Route) => ({ ...prev, case: run.case }));
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
        "transition-colors duration-fast ease-motion-out",
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
