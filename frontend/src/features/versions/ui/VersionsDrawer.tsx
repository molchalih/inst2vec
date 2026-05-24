import { useCallback } from "react";
import { useAtom, useAtomValue } from "jotai";
import { caseAtom, manifestAtom, useIsIntroPlaying } from "@/state";
import { useEscKey } from "@/interaction";
import { drawerOpenAtom } from "../state.atom";
import { VersionPill } from "./VersionPill";

/**
 * Version drawer: collapsed-by-default, top-anchored. The tongue is
 * the only thing visible until clicked. Esc closes when open. No
 * close-on-outside-click — canvas pans would close it accidentally.
 *
 * Geometry: the drawer panel sits inside an absolutely-positioned
 * full-width container at top: 0. When closed, the panel is translated
 * up by 100% so it sits entirely above the viewport. The tongue is
 * positioned at the panel's bottom edge (top: drawer-h), so when the
 * panel is translated up it slides with the panel and ends up at
 * y = 0 (just visible at the top of the viewport). When open, the
 * panel translates back to 0 and the tongue ends up at y = drawer-h
 * (hugging the bottom of the open drawer).
 */
export const VersionsDrawer = () => {
  const manifest = useAtomValue(manifestAtom);
  const activeCase = useAtomValue(caseAtom);
  const introPlaying = useIsIntroPlaying();
  const [open, setOpen] = useAtom(drawerOpenAtom);

  // Locked shut until the entrance flight settles: opening mid-flight would
  // overlay the pill row on dots still streaming to their positions.
  const toggle = useCallback((): void => {
    if (introPlaying) return;
    setOpen((v) => !v);
  }, [introPlaying, setOpen]);
  useEscKey(open, () => setOpen(false));

  const activeRun = manifest?.runs.find((r) => r.case === activeCase);
  const activeLabel = activeRun ? activeRun.label : "";

  return (
    <div
      className={[
        "absolute inset-x-0 top-0 z-40 pointer-events-none",
        "transform transition-transform duration-medium ease-motion-out",
        open ? "translate-y-0" : "-translate-y-drawer-h",
      ].join(" ")}
    >
      {/* The bar inherits pointer-events-none from the container: its
          full width would otherwise swallow clicks on whatever sits
          under the top strip (notably the inspector panel's close ×).
          Only the pills opt back into pointer events. */}
      <div
        className={[
          "h-drawer-h w-full",
          "flex items-center justify-center gap-drawer-gap px-drawer-px",
        ].join(" ")}
      >
        {manifest?.runs.map((run) => (
          <VersionPill key={run.id} run={run} />
        ))}
      </div>
      <button
        type="button"
        onClick={toggle}
        disabled={introPlaying}
        aria-expanded={open}
        aria-label="Toggle version drawer"
        className={[
          "absolute left-1/2 -translate-x-1/2 top-full pt-tongue-pt",
          "flex flex-col items-center text-xs text-fg-muted",
          "pointer-events-auto",
          "transition-colors duration-fast ease-motion-out",
          "disabled:opacity-50 disabled:cursor-not-allowed",
        ].join(" ")}
      >
        <span aria-hidden className="leading-none text-base text-fg-default">
          {open ? "−" : "+"}
        </span>
        {!open && <span className="leading-none mt-1">{activeLabel}</span>}
      </button>
    </div>
  );
};
