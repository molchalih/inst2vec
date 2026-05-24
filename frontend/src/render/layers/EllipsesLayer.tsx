import { useCallback, useMemo } from "react";
import type { Graphics as PixiGraphics } from "pixi.js";
import {
  useStretchedRun, useViewport, useTransition, useIntro,
} from "@/state";
import {
  joinClustersById, interpolateEllipses,
  ellipseAlphaScale, ellipseSide, introEllipseAlpha,
} from "@/core";
import { tokens } from "@/ui/tokens";
import { drawEllipses } from "../draw/drawEllipses";
import { runToEllipsesFrame, type EllipsesFrame, type DrawableEllipse } from "../frame";

export const EllipsesLayer = () => {
  const run = useStretchedRun();
  const transition = useTransition();
  const intro = useIntro();
  const [viewport] = useViewport();

  const frame = useMemo<EllipsesFrame>(() => {
    if (transition) {
      const joined = joinClustersById(transition.from, transition.to);
      const side = ellipseSide(transition.phase);
      const ellipses: DrawableEllipse[] = interpolateEllipses(joined, side, tokens.palette.cluster, tokens.palette.noise)
        .map((e) => ({
          id: e.id,
          cx: e.cx, cy: e.cy, rx: e.rx, ry: e.ry, angle: e.angle,
          color: e.color,
        }));
      return {
        ellipses,
        alphaScale: ellipseAlphaScale(transition.phase, transition.progress),
        strokeWidthScale: 1,
      };
    }
    const base = runToEllipsesFrame(run);
    if (intro) {
      return { ...base, alphaScale: introEllipseAlpha(intro.phase, intro.progress) };
    }
    return base;
  }, [run, transition, intro]);

  const draw = useCallback(
    (g: PixiGraphics) => { drawEllipses(g, frame, viewport); },
    [frame, viewport],
  );

  return <pixiGraphics draw={draw} zIndex={0} />;
};
