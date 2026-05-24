import { useLayoutEffect, useRef } from "react";
import { useAtomValue, useSetAtom, useStore } from "jotai";
import {
  selectionAtom, stretchedRunAtom, visibleRectAtom, viewportAtom,
  useFocusViewport,
  isTransitioningAtom, introAtom,
  type Selection,
} from "@/state";
import {
  easeOutCubic, fitBoundsToRect, focusTransform,
  type Transform, type Rect,
} from "@/core";
import { tokens } from "@/ui/tokens";
import type { AtlasRun } from "@/data";

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
const lerpTransform = (a: Transform, b: Transform, t: number): Transform => ({
  x: lerp(a.x, b.x, t),
  y: lerp(a.y, b.y, t),
  scale: lerp(a.scale, b.scale, t),
});

const computeTarget = (
  sel: Selection,
  run: AtlasRun,
  rect: Rect,
): Transform | null => {
  if (!sel) return null;
  // Reference scale = the run's natural fit inside the (panel-inset)
  // rect. It's the basis for both the focus zoom band and clampPanZoom,
  // so focusTransform caps the zoom to exactly what the clamp allows and
  // keeps the focused point centred in the visible rect. Using the rect
  // (not the full viewport) makes the chosen scale invariant to the panel.
  const fitScale = fitBoundsToRect(
    run.bounds, rect, tokens.interaction.focus.runFitPadding,
  ).scale;
  const limits = tokens.viewport.clamp;

  if (sel.kind === "cluster") {
    const c = run.clusters.find((c) => c.id === sel.clusterId);
    if (!c) return null;
    const b = {
      minX: c.cx - c.rx, maxX: c.cx + c.rx,
      minY: c.cy - c.ry, maxY: c.cy + c.ry,
    };
    const desired = fitBoundsToRect(
      b, rect, tokens.interaction.focus.clusterFitPadding,
    ).scale;
    return focusTransform(
      { x: (b.minX + b.maxX) / 2, y: (b.minY + b.maxY) / 2 },
      rect, desired, fitScale, limits,
    );
  }
  // creator: zoom creatorScaleFactor× past the run fit, centred on the dot.
  const u = run.users.find(([id]) => id === sel.creatorId);
  if (!u) return null;
  return focusTransform(
    { x: u[1], y: u[2] },
    rect,
    fitScale * tokens.interaction.focus.creatorScaleFactor,
    fitScale, limits,
  );
};

/**
 * Watches selectionAtom. On a fresh selection (or visibleRect change
 * while selected), eases viewportAtom from its current value to the
 * computed target over tokens.motion.cameraFocus.durationMs using
 * easeOutCubic. On non-null→null transitions, eases back to the
 * run's natural fit and clears the user override on landing. A
 * running intro or version-switch yields without writing.
 */
export const useCameraFocus = (): void => {
  const selection = useAtomValue(selectionAtom);
  const run = useAtomValue(stretchedRunAtom);
  const rect = useAtomValue(visibleRectAtom);
  const isTransitioning = useAtomValue(isTransitioningAtom);
  const intro = useAtomValue(introAtom);
  const setViewport = useSetAtom(viewportAtom);
  const setFocusViewport = useFocusViewport();
  const store = useStore();
  const rafRef = useRef<number | null>(null);
  // Tracks the last seen selection so we can detect non-null→null
  // (deselect) and ease back to the fit instead of snapping.
  const prevSelRef = useRef<Selection>(null);

  useLayoutEffect(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    const prev = prevSelRef.current;
    prevSelRef.current = selection;
    if (!run || isTransitioning || intro !== null) return;
    if (rect.width <= 0 || rect.height <= 0) return;

    const deselected = prev !== null && selection === null;
    const target = selection
      ? computeTarget(selection, run, rect)
      : deselected
        ? fitBoundsToRect(run.bounds, rect, tokens.interaction.focus.runFitPadding)
        : null;
    if (!target) return;

    // Snapshot start once per ease via store.get (no subscription —
    // viewport changes during the ease won't restart it).
    const start = store.get(viewportAtom);
    const startTime = performance.now();
    const duration = tokens.motion.cameraFocus.durationMs;

    const tick = (now: number): void => {
      const elapsed = now - startTime;
      const t = duration <= 0 ? 1 : Math.min(elapsed / duration, 1);
      const k = easeOutCubic(t);
      // Write raw frames (focusViewportAtom does not clamp): a tween between
      // two pre-validated endpoints must not be re-clamped against the
      // current fit band, which shifts when selection insets/uninsets the
      // visible rect and would otherwise pin a frame near `start`.
      setFocusViewport(lerpTransform(start, target, k));
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        rafRef.current = null;
        // On deselect, hand the viewport back to the derived fit so
        // future size/case changes re-fit automatically.
        if (deselected) setViewport(null);
      }
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [selection, run, rect, isTransitioning, intro, setViewport, setFocusViewport, store]);
};
