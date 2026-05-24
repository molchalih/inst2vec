import { atom, useAtom, useSetAtom } from "jotai";
import {
  applyWheel,
  clampPanZoom,
  fitBoundsToRect,
  type Transform,
  type Vec2,
} from "@/core";
import { tokens } from "@/ui/tokens";
import { stretchedRunAtom } from "./stretched-run.atom";
import { viewportSizeAtom } from "./viewport-size.atom";
import { visibleRectAtom } from "./visible-rect.atom";

const IDENTITY: Transform = { x: 0, y: 0, scale: 1 };

/**
 * Optional override on top of the derived fit. null = use the derived
 * fit. Pan/zoom writes a Transform; the case-switch ease writes lerp
 * frames and clears back to null when it lands.
 *
 * Module-private: consumers go through viewportAtom below. The fit /
 * override split is an implementation detail.
 */
const userViewportAtom = atom<Transform | null>(null);

/**
 * Synchronously-derived fit for the current (stretched run, viewport
 * size). Returns IDENTITY when either input isn't ready — this value
 * is never rendered because Stage gates the Pixi root on run
 * availability.
 */
const fittedViewportAtom = atom<Transform>((get) => {
  const run = get(stretchedRunAtom);
  const size = get(viewportSizeAtom);
  const rect = get(visibleRectAtom);
  if (!run || size.width <= 0 || size.height <= 0 || rect.width <= 0 || rect.height <= 0) {
    return IDENTITY;
  }
  return fitBoundsToRect(run.bounds, rect, tokens.interaction.focus.runFitPadding);
});

/**
 * Public viewport surface.
 *
 * Read: override if set, else the derived fit. Synchronous — the
 * first frame consumers see is already correct, so there is no
 * "render at identity, then snap" timing window.
 *
 * Write: setViewport(t) sets the override; setViewport(prev => f(prev))
 * updates it; setViewport(null) clears it back to the derived fit.
 * Every non-null write is clamped via clampPanZoom — wheel, drag, and
 * ease-frame writes all pass through the same chokepoint.
 */
export const viewportAtom = atom<
  Transform,
  [Transform | null | ((curr: Transform) => Transform)],
  void
>(
  (get) => get(userViewportAtom) ?? get(fittedViewportAtom),
  (get, set, next) => {
    if (next === null) {
      set(userViewportAtom, null);
      return;
    }
    const curr = get(viewportAtom);
    const raw = typeof next === "function" ? next(curr) : next;
    const run = get(stretchedRunAtom);
    const size = get(viewportSizeAtom);
    if (!run || size.width <= 0 || size.height <= 0) {
      set(userViewportAtom, raw);
      return;
    }
    const fit = get(fittedViewportAtom);
    set(userViewportAtom, clampPanZoom(raw, run.bounds, size, fit.scale, tokens.viewport.clamp));
  },
);

export const useViewport = () => useAtom(viewportAtom);

/**
 * Camera-tween write path. The ease in useCameraFocus writes its
 * interpolated frames here, and they pass through verbatim — no band clamp.
 *
 * A tween runs between two pre-validated endpoints (start = the current
 * viewport; target = focusTransform's capped, centred result, or the
 * derived fit on deselect), so its frames are deliberate motion, not user
 * input. Clamping each frame against the *current* fit band is wrong:
 * selecting/deselecting shifts the band (visibleRect insets/uninsets →
 * fitScale jumps), so frames near `start` — captured under the previous
 * band — land outside the new one and get pinned. A large cluster focuses
 * below the full-viewport fit floor; clamping each deselect frame up to
 * that floor snapped the scale instead of easing it. Free pan/zoom still
 * clamp via viewportAtom / wheelZoomAtom — gestures and the resting state
 * are where the band belongs.
 */
export const focusViewportAtom = atom<null, [Transform], void>(
  null,
  (_get, set, next) => {
    set(userViewportAtom, next);
  },
);

export const useFocusViewport = () => useSetAtom(focusViewportAtom);

/**
 * Wheel zoom anchored on the cursor. Goes through this dedicated path
 * (rather than `viewportAtom`) so the scale clamp is applied *before*
 * the translation is computed — otherwise the world point under the
 * cursor drifts when the gesture is at the zoom ceiling or floor.
 */
export const wheelZoomAtom = atom<null, [{ cursor: Vec2; factor: number }], void>(
  null,
  (get, set, { cursor, factor }) => {
    const curr = get(viewportAtom);
    const run = get(stretchedRunAtom);
    const size = get(viewportSizeAtom);
    if (!run || size.width <= 0 || size.height <= 0) {
      set(userViewportAtom, { ...curr, scale: curr.scale * factor });
      return;
    }
    const fit = get(fittedViewportAtom);
    const scaleBounds = {
      min: tokens.viewport.clamp.minScaleFactor * fit.scale,
      max: tokens.viewport.clamp.maxScaleFactor * fit.scale,
    };
    const next = applyWheel(curr, cursor, factor, scaleBounds);
    set(userViewportAtom, clampPanZoom(next, run.bounds, size, fit.scale, tokens.viewport.clamp));
  },
);

export const useWheelZoom = () => useSetAtom(wheelZoomAtom);
