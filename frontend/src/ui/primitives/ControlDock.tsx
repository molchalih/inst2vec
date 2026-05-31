import type { ReactNode } from "react";

/**
 * Invisible right-edge control dock. Pins its children to the canvas's right
 * edge, bottom-anchored, in a vertical column so controls stack upward as the
 * chrome grows. Position is stable regardless of inspector/panel state — the
 * dock owns placement; the controls it hosts are position-agnostic glyphs.
 * A token gap separates stacked items. Pointer events pass through the empty
 * dock area; only the hosted controls capture them.
 */
export const ControlDock = ({ children }: { children: ReactNode }) => (
  <div
    className={[
      "pointer-events-none",
      "absolute bottom-dock-offset right-dock-offset",
      "flex flex-col-reverse items-end gap-dock-gap",
    ].join(" ")}
  >
    {children}
  </div>
);
