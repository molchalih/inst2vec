import type { AlphaFilter, Graphics } from "pixi.js";
import type { Transform } from "@/core";
import { tokens } from "@/ui/tokens";
import type { DotsFrame } from "../frame";

export const drawDots = (
  g: Graphics,
  frame: DotsFrame,
  viewport: Transform,
  groupFade?: AlphaFilter | null,
): void => {
  g.clear();
  // True group fade. `g.alpha` (worldAlpha) is NOT a post-composite opacity in
  // Pixi — it folds into each circle's source alpha within the single batched
  // draw call, so the ~hundreds of dots stacked on the centre point during the
  // intro fade still composite over one another and saturate to opaque almost
  // immediately (identical to a per-dot multiply). An AlphaFilter renders the
  // layer to an offscreen target and applies the alpha once, post-composite —
  // a real group opacity with no overdraw. Attached only while fading
  // (alphaScale < 1); otherwise the layer draws straight, with no extra pass.
  if (groupFade && frame.alphaScale < 1) {
    groupFade.alpha = frame.alphaScale;
    g.filters = [groupFade];
  } else if (g.filters) {
    g.filters = [];
  }
  const base =
    (tokens.dot.radius * frame.radiusScale) /
    Math.sqrt(Math.max(viewport.scale, 1e-6));
  for (const u of frame.users) {
    g.circle(u.x, u.y, base * u.radiusScale)
      .fill({ color: u.color, alpha: u.alpha });
  }
};
