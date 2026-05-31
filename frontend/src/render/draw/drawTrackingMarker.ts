import type { Graphics } from "pixi.js";
import type { Transform } from "@/core";
import { tokens } from "@/ui/tokens";

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

/**
 * Tracked-dot beacon border. The base dot (drawn by DotsLayer) is the
 * normal-size cluster-colour core; this adds ONLY a thin white ring around it
 * at `dot.radius + marker.gapPx`, whose alpha breathes on the caller's sine
 * `pulse` (0..1, floored at `borderAlphaMin` so it never blinks fully out) and
 * is scaled by `vis` (0..1) so the whole beacon fades in on track / out on
 * untrack. Screen-compensated with the same invSqrtScale the dot/hover draws
 * use. Renders ABOVE the base dots layer so the ring isn't occluded by
 * neighbouring dots. Draws nothing when `pos` is null or `vis` is zero (tracked
 * creator absent / fully faded out).
 */
export const drawTrackingMarker = (
  g: Graphics,
  pos: readonly [number, number] | null,
  viewport: Transform,
  pulse: number,
  vis: number,
): void => {
  g.clear();
  if (!pos || vis <= 0) return;
  const [x, y] = pos;
  const { gapPx, borderWidth, borderAlphaMin, borderAlphaMax } = tokens.track.marker;
  const borderColor = tokens.dot.strokeColorHover;

  const invSqrtScale = 1 / Math.sqrt(Math.max(viewport.scale, 1e-6));
  const r = (tokens.dot.radius + gapPx) * invSqrtScale;
  const borderAlpha = lerp(borderAlphaMin, borderAlphaMax, pulse) * vis;

  g.circle(x, y, r).stroke({
    color: borderColor,
    alpha: borderAlpha,
    width: borderWidth * invSqrtScale,
  });
};
