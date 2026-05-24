import type { Graphics } from "pixi.js";
import { ellipsePoints, colorForCluster, type Transform } from "@/core";
import type { AtlasRun } from "@/data";
import type { CrossfadeSlot } from "@/interaction";
import { tokens } from "@/ui/tokens";

const ELLIPSE_SEGMENTS = 64;
const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

/**
 * Cluster hover overlay. Draws every supplied slot in order, each fading
 * from invisible (pulse=0) to highlighted stroke+fill (pulse=1). During
 * an A→B handoff the list carries both A (pulse falling) and B (pulse
 * rising), so the two ellipses cross-fade in parallel.
 */
export const drawHoverCluster = (
  g: Graphics,
  slots: ReadonlyArray<CrossfadeSlot>,
  run: AtlasRun | null,
  viewport: Transform,
): void => {
  g.clear();
  if (!run) return;
  for (const slot of slots) {
    if (slot.pulse <= 0) continue;
    const c = run.clusters.find((cc) => cc.id === slot.id);
    if (!c) continue;
    const color = colorForCluster(c.id, tokens.palette.cluster, tokens.palette.noise);
    const width =
      lerp(tokens.ellipse.strokeWidth, tokens.ellipse.strokeWidthHover, slot.pulse) /
      Math.max(viewport.scale, 1e-6);
    const strokeAlpha = lerp(0, tokens.ellipse.strokeAlphaHover, slot.pulse);
    const fillAlpha = lerp(0, tokens.ellipse.fillAlpha, slot.pulse);
    const pts = ellipsePoints(c, ELLIPSE_SEGMENTS).flatMap((p) => [p.x, p.y]);
    g.poly(pts)
      .fill({ color, alpha: fillAlpha })
      .stroke({ color, alpha: strokeAlpha, width });
  }
};
