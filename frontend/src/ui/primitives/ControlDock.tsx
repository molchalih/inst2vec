import { Children, isValidElement, type ReactNode } from "react";
import { tokens } from "../tokens";

/**
 * One hosted control plus its entrance. The dock is absent until the intro
 * flight settles; each item then slides in from the right edge, offset by
 * `index * staggerMs` so a growing column arrives one glyph at a time.
 * `inert` (not just hidden) keeps a not-yet-revealed control out of the
 * focus order and pointer reach; reduced-motion users get the resting state
 * with no slide.
 */
const DockItem = ({
  index, revealed, children,
}: { index: number; revealed: boolean; children: ReactNode }) => (
  <div
    inert={revealed ? undefined : true}
    className={[
      "transition-[transform,opacity] duration-chrome ease-chrome",
      "motion-reduce:transition-none",
      revealed ? "translate-x-0 opacity-100" : "translate-x-dock-reveal-x opacity-0",
    ].join(" ")}
    style={{ transitionDelay: revealed ? `${index * tokens.motion.chrome.staggerMs}ms` : "0ms" }}
  >
    {children}
  </div>
);

/**
 * Invisible right-edge control dock. Pins its children to the canvas's right
 * edge, bottom-anchored, in a vertical column so controls stack upward as the
 * chrome grows. Position is stable regardless of inspector/panel state — the
 * dock owns placement; the controls it hosts are position-agnostic glyphs.
 * A token gap separates stacked items. Pointer events pass through the empty
 * dock area; only the hosted controls capture them. `revealed` drives the
 * one-time chrome entrance (see DockItem).
 */
export const ControlDock = ({
  revealed, children,
}: { revealed: boolean; children: ReactNode }) => (
  <div
    className={[
      "pointer-events-none",
      "absolute bottom-dock-offset right-dock-offset",
      "flex flex-col-reverse items-end gap-dock-gap",
    ].join(" ")}
  >
    {Children.toArray(children).map((child, index) => (
      <DockItem
        key={isValidElement(child) && child.key !== null ? child.key : index}
        index={index}
        revealed={revealed}
      >
        {child}
      </DockItem>
    ))}
  </div>
);
