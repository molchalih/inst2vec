import type { Graphics } from "pixi.js";
import type { Transform } from "@/core";
import { tokens } from "@/ui/tokens";

/**
 * Persistent tracked-dot beacon halo. A soft white radial glow that bleeds
 * outward from the tracked dot, drawn as `falloffSteps` concentric white fills
 * whose alpha decays toward the edge. Its outer radius AND overall alpha breathe
 * together on the caller's sine `pulse` (0..1), so the halo reads as a living
 * beacon rather than a blinking ring. Its overall alpha is scaled by `vis`
 * (0..1) so the halo fades in on track / out on untrack in lockstep with the
 * border. Screen-compensated with the same invSqrtScale the dot/hover draws
 * use. Renders BENEATH the dots layer, so the dot's cluster-colour core stays
 * crisp on top. Draws nothing when `pos` is null or `vis` is zero (tracked
 * creator absent / fully faded out).
 */
export const drawTrackingHalo = (
  g: Graphics,
  pos: readonly [number, number] | null,
  viewport: Transform,
  pulse: number,
  vis: number,
): void => {
  g.clear();
  if (!pos || vis <= 0) return;
  const [x, y] = pos;
  const { radiusPx, falloffSteps, alphaMin, alphaMax } = tokens.track.halo;
  const color = tokens.dot.strokeColorHover;

  const invSqrtScale = 1 / Math.sqrt(Math.max(viewport.scale, 1e-6));
  // Radius and alpha breathe together: at the pulse trough the glow is both
  // smaller and fainter, at the crest larger and brighter.
  const alpha = (alphaMin + (alphaMax - alphaMin) * pulse) * vis;
  const radiusFloor = alphaMin / alphaMax; // crest:trough radius ratio == alpha ratio
  const outerR = radiusPx * (radiusFloor + (1 - radiusFloor) * pulse) * invSqrtScale;

  // Concentric fills from outermost (faintest) inward (brightest core glow),
  // each step adding alpha so the centre reads as a lit point and the edge
  // fades to nothing — a soft radial falloff approximated in falloffSteps rings.
  for (let i = 0; i < falloffSteps; i++) {
    const inner = (i + 1) / falloffSteps; // 1/n .. 1 (innermost fill largest share)
    const r = outerR * (1 - i / falloffSteps);
    g.circle(x, y, r).fill({ color, alpha: alpha * inner / falloffSteps });
  }
};
