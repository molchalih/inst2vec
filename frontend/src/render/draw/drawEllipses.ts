import type { Graphics } from "pixi.js";
import { ellipsePoints, type Transform } from "@/core";
import { tokens } from "@/ui/tokens";
import type { EllipsesFrame } from "../frame";

const SEGMENTS = 64;

export const drawEllipses = (
  g: Graphics,
  frame: EllipsesFrame,
  viewport: Transform,
): void => {
  g.clear();
  const width =
    (tokens.ellipse.strokeWidth * frame.strokeWidthScale) /
    Math.max(viewport.scale, 1e-6);
  for (const e of frame.ellipses) {
    const pts = ellipsePoints(e, SEGMENTS).flatMap((p) => [p.x, p.y]);
    g.poly(pts)
      .fill({ color: e.color, alpha: tokens.ellipse.fillAlpha * frame.alphaScale })
      .stroke({
        color: e.color,
        alpha: tokens.ellipse.strokeAlpha * frame.alphaScale,
        width,
      });
  }
};
