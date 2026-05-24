import type { Graphics } from "pixi.js";
import type { Transform } from "@/core";
import { tokens } from "@/ui/tokens";
import type { DotsFrame } from "../frame";

export const drawDots = (
  g: Graphics,
  frame: DotsFrame,
  viewport: Transform,
): void => {
  g.clear();
  const base =
    (tokens.dot.radius * frame.radiusScale) /
    Math.sqrt(Math.max(viewport.scale, 1e-6));
  for (const u of frame.users) {
    g.circle(u.x, u.y, base * u.radiusScale)
      .fill({ color: u.color, alpha: u.alpha * frame.alphaScale });
  }
};
