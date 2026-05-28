import { useRef, type CSSProperties } from "react";
import { useAtomValue } from "jotai";
import {
  activeRunAtom,
  inspectorPhaseAtom,
  isContentMounted,
  isPanelOpen,
  selectionAtom,
  shouldAdvanceDisplayed,
  useClearSelection,
  type InspectorPhase,
  type Selection,
} from "@/state";
import { colorForCluster } from "@/core";
import { Panel } from "@/ui";
import { tokens } from "@/ui/tokens";
import { ClusterPane } from "./panes/ClusterPane";
import { CreatorPane } from "./panes/CreatorPane";
import { PaneShell } from "./ui/PaneShell";
import { useInspectorChoreography } from "./useInspectorChoreography";

type ActiveSelection = NonNullable<Selection>;

/**
 * Inspector container. Owns the choreographed open / close / swap
 * sequence (delegated to `useInspectorChoreography`) and renders the
 * appropriate pane for the current selection.
 *
 * Three derived values, three different timings:
 *
 * - **`displayed`** — the selection backing the rendered pane. Lags
 *   `selectionAtom` during the closing-content sub-phase so the old
 *   pane finishes its slide-out before the new one mounts. Resolved
 *   synchronously during render via a ref (no `useEffect` delay → no
 *   pane-swap flicker).
 * - **`accentSel`** — the selection driving `--accent`. Tracks
 *   `selectionAtom` directly, so the colour starts morphing the
 *   instant the user clicks a new dot and runs in parallel with the
 *   closing-content slide.
 * - **`contentClass`** — `inspector-content--in` / `--out` based on
 *   the phase; the CSS keyframes do the actual translate + fade.
 */
export const Inspector = () => {
  useInspectorChoreography();
  const sel = useAtomValue(selectionAtom);
  const phase = useAtomValue(inspectorPhaseAtom);
  const run = useAtomValue(activeRunAtom);
  const close = useClearSelection();
  const motion = tokens.inspector.motion;

  const displayed = useDeferredSelection(sel, phase);
  if (!displayed) return null;

  // Accent leads the swap so the new colour is fully resolved by the
  // time the new pane slides in. Falls back to `displayed` during the
  // closing-slide window when `sel` is already null.
  const accentSel: ActiveSelection = sel ?? displayed;
  const accentClusterId = accentSel.kind === "cluster"
    ? accentSel.clusterId
    : run?.users.find(([id]) => id === accentSel.creatorId)?.[3];
  const accent = colorForCluster(
    accentClusterId ?? -1,
    tokens.palette.cluster,
    tokens.palette.noise,
  );
  const contentClass = contentClassFor(phase);

  return (
    <Panel
      open={isPanelOpen(phase)}
      side="left"
      onClose={close}
      durationMs={motion.slideMs}
    >
      <div
        className="inspector-accent-host"
        style={{
          "--accent": accent,
          "--accent-morph-ms": `${motion.accentMs}ms`,
          "--content-slide-ms": `${motion.contentMs}ms`,
          ...stage,
        } as CSSProperties}
      >
        <div aria-hidden="true" style={washStyle} />
        <PaneShell onClose={close}>
          {isContentMounted(phase) && (
            <div className={`inspector-content ${contentClass}`}>
              {displayed.kind === "cluster"
                ? <ClusterPane clusterId={displayed.clusterId} />
                : <CreatorPane creatorId={displayed.creatorId} />}
            </div>
          )}
        </PaneShell>
      </div>
    </Panel>
  );
};

/**
 * Resolve which selection's pane should be visible right now, based on
 * the current phase. The last-shown selection is held in a ref so it
 * persists through closing phases without a `useEffect` delay.
 *
 * Crucially, the cache only advances during entering sub-phases (see
 * `shouldAdvanceDisplayed`). Excluding the steady `open` phase
 * eliminates a single-frame race during reselection: the click flips
 * `selectionAtom` synchronously but the choreography setPhase fires
 * after render, so an "open"-permissive predicate would snapshot the
 * new selection one frame too early and the new pane would flash in
 * before the close animation began.
 *
 * Writes to the ref happen in render and are idempotent for any given
 * `(phase, sel)` pair, which is the supported pattern for caching
 * derived state in React.
 */
const useDeferredSelection = (
  sel: Selection,
  phase: InspectorPhase,
): ActiveSelection | null => {
  const ref = useRef<ActiveSelection | null>(null);
  if (sel !== null && shouldAdvanceDisplayed(phase)) {
    ref.current = sel;
  } else if (phase === "closed") {
    ref.current = null;
  }
  return ref.current;
};

/**
 * Map a phase to the CSS animation class on the content wrapper.
 *
 * - `opening-content` / `open` → `--in`  (translate `-100% → 0`, fade in)
 * - `closing-content` / `closing-slide` → `--out` (translate `0 → -100%`,
 *   fade out). `closing-slide` keeps the class so the animation runs
 *   to completion alongside the panel slide-out.
 */
const contentClassFor = (phase: InspectorPhase): string => {
  if (phase === "closing-content" || phase === "closing-slide") {
    return "inspector-content--out";
  }
  return "inspector-content--in";
};

// Fills the panel and is the containing block for the wash layer.
const stage: CSSProperties = {
  position: "relative", flex: 1, minHeight: 0,
  display: "flex", flexDirection: "column",
};

// Full-bleed top→bottom accent wash behind the content. Reads
// `--accent` (registered via `@property` in index.html) so the
// gradient morphs smoothly when the parent transitions the variable
// via the `.inspector-accent-host` class.
const washStyle: CSSProperties = {
  position: "absolute", inset: 0, pointerEvents: "none",
  background: `linear-gradient(to bottom, color-mix(in srgb, var(--accent) ${tokens.inspector.wash.topPct}%, transparent), transparent ${tokens.inspector.wash.fadeStop}%)`,
};
