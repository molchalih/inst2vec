import type { Graphics } from "pixi.js";
import { colorForCluster, type Transform } from "@/core";
import type { AtlasRun } from "@/data";
import { tokens } from "@/ui/tokens";

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

/**
 * Dot hover overlay. Fades from invisible (pulse=0) to a slightly
 * larger pulsed dot (pulse=1). The base dot is still drawn by
 * DotsLayer underneath, so at pulse=0 the visual is unchanged.
 */
export const drawHoverDot = (
  g: Graphics,
  dotId: number | null,
  run: AtlasRun | null,
  viewport: Transform,
  pulse: number,
): void => {
  g.clear();
  if (!run || dotId === null || pulse <= 0) return;
  const u = run.users.find(([id]) => id === dotId);
  if (!u) return;
  const [, x, y, clusterId] = u;
  const invSqrtScale = 1 / Math.sqrt(Math.max(viewport.scale, 1e-6));
  const baseR = tokens.dot.radius * invSqrtScale;
  const hoverR = tokens.dot.radiusHover * invSqrtScale;
  const r = lerp(baseR, hoverR, pulse);
  const color = colorForCluster(clusterId, tokens.palette.cluster, tokens.palette.noise);
  const strokeWidth = tokens.dot.strokeWidthHover * invSqrtScale;
  g.circle(x, y, r)
    .fill({ color, alpha: pulse })
    .stroke({ color: tokens.dot.strokeColorHover, alpha: pulse, width: strokeWidth });
};
