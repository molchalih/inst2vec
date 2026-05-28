import { useEffect, useRef } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import { inspectorPhaseAtom, selectionAtom, type Selection } from "@/state";
import { tokens } from "@/ui/tokens";

/**
 * Identity key over a Selection. `null` and matching key both mean
 * "no transition" — only key-level changes drive a phase sequence.
 */
const keyOf = (sel: Selection): string | null => {
  if (sel === null) return null;
  return sel.kind === "cluster"
    ? `cluster:${sel.clusterId}`
    : `creator:${sel.creatorId}`;
};

/**
 * Drives `inspectorPhaseAtom` from `selectionAtom` transitions.
 *
 * Three sequences, all reusing the timing in `tokens.inspector.motion`:
 *
 * - **open**  (`null → A`): `opening-slide → opening-content → open`
 * - **close** (`A → null`): `closing-content → closing-slide → closed`
 * - **swap**  (`A → B`):    `closing-content → opening-content → open`
 *   (panel stays mounted — only the content wrapper animates).
 *
 * Mount semantics: on first mount with no selection the phase stays at
 * `closed`. Cleanup clears all in-flight timers, so any rapid input
 * cancels the previous sequence cleanly.
 */
export const useInspectorChoreography = (): void => {
  const sel = useAtomValue(selectionAtom);
  const setPhase = useSetAtom(inspectorPhaseAtom);
  const prevKey = useRef<string | null>(keyOf(sel));
  const mounted = useRef(false);

  useEffect(() => {
    if (!mounted.current) {
      mounted.current = true;
      prevKey.current = keyOf(sel);
      // Deep-link hydration: selection is already set on first mount.
      // Pass through `opening-content` so `useDeferredSelection` (which
      // only advances its ref on entering sub-phases) snapshots the
      // selection, then settle at `open` on the next tick. Without the
      // intermediate phase the snapshot never happens and the inspector
      // body stays blank on first paint for deep-link URLs.
      if (sel !== null) {
        setPhase("opening-content");
        const settle = window.setTimeout(() => setPhase("open"), 0);
        return () => window.clearTimeout(settle);
      }
      return;
    }
    const prev = prevKey.current;
    const next = keyOf(sel);
    if (prev === next) return;
    prevKey.current = next;

    const motion = tokens.inspector.motion;
    const slide = motion.slideMs;
    const content = motion.contentMs;
    // Stages start at `(1 - overlap) * firstStageDuration`, so the
    // second stage begins while the first is still mid-flight.
    // Applied only to open + close (panel + content are distinct DOM
    // elements with independent animations). Swap stays sequential.
    const slideHandoff = slide * (1 - motion.overlap);
    const contentHandoff = content * (1 - motion.overlap);
    const timers: number[] = [];

    if (prev === null && next !== null) {
      // Open: panel slide → (overlap) content slide → settled.
      setPhase("opening-slide");
      timers.push(window.setTimeout(() => setPhase("opening-content"), slideHandoff));
      timers.push(window.setTimeout(() => setPhase("open"), slideHandoff + content));
    } else if (prev !== null && next === null) {
      // Close: content slide → (overlap) panel slide → unmount.
      setPhase("closing-content");
      timers.push(window.setTimeout(() => setPhase("closing-slide"), contentHandoff));
      timers.push(window.setTimeout(() => setPhase("closed"), contentHandoff + slide));
    } else {
      // Swap: content out, then content in. Sequential — the two
      // halves animate the same DOM element and would clash if
      // overlapped. `Inspector` watches for the `opening-content`
      // boundary to swap the displayed selection.
      setPhase("closing-content");
      timers.push(window.setTimeout(() => setPhase("opening-content"), content));
      timers.push(window.setTimeout(() => setPhase("open"), 2 * content));
    }
    return () => timers.forEach((t) => window.clearTimeout(t));
  }, [sel, setPhase]);
};
