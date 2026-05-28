import { atom } from "jotai";

/**
 * Two-stage inspector open/close choreography. Owned by `Inspector.tsx`
 * (sole writer via `useInspectorChoreography`); read by `Panel` and
 * the pane-content wrapper.
 *
 * Sequence on open (`closed → open`):
 *   1. `opening-slide`    — panel slides in, pane content not mounted.
 *   2. `opening-content`  — content mounts and slides in from the left.
 *   3. `open`             — steady state.
 *
 * Sequence on close (`open → closed`):
 *   1. `closing-content`  — content slides out toward the right.
 *   2. `closing-slide`    — content unmounts, panel slides out leftward.
 *   3. `closed`           — fully unmounted.
 *
 * Swap (`A → B` while open) reuses `closing-content` → `opening-content`
 * with no slide sub-phases; the panel stays mounted.
 */
export type InspectorPhase =
  | "closed"
  | "opening-slide"
  | "opening-content"
  | "open"
  | "closing-content"
  | "closing-slide";

export const inspectorPhaseAtom = atom<InspectorPhase>("closed");

/** True when the panel chrome should be mounted (kept on for content). */
export const isPanelOpen = (phase: InspectorPhase): boolean =>
  phase !== "closed" && phase !== "closing-slide";

/**
 * True when the pane-content wrapper should be mounted.
 *
 * Includes `closing-slide` so the content's `--out` animation can keep
 * running in parallel with the panel slide-out — that overlap is the
 * whole point of staging the close path. The content unmounts only
 * when phase reaches `closed`.
 */
export const isContentMounted = (phase: InspectorPhase): boolean =>
  phase === "opening-content" ||
  phase === "open" ||
  phase === "closing-content" ||
  phase === "closing-slide";

/**
 * True when the consumer should advance its "displayed selection" cache
 * to the current `selectionAtom` value.
 *
 * **Deliberately excludes `open`.** When the user clicks a new dot
 * while the inspector is in steady `open`, `selectionAtom` flips
 * synchronously but the choreography effect fires only after render.
 * If `open` were included, the next render would snapshot the new
 * selection BEFORE the phase transitioned to `closing-content`, and
 * the user would see the new pane content briefly flash in place
 * before the slide-out animation began. Snapshotting only on entering
 * sub-phases (`opening-slide`, `opening-content`) makes the cache
 * advance in lockstep with the visible "show new content" transition.
 */
export const shouldAdvanceDisplayed = (phase: InspectorPhase): boolean =>
  phase === "opening-slide" || phase === "opening-content";
