import { useCallback, useLayoutEffect, useRef } from "react";
import type { Graphics as PixiGraphics } from "pixi.js";
import { useAtomValue } from "jotai";
import { hoverAtom, stretchedRunAtom, viewportAtom, useIsTransitioning } from "@/state";
import { easeOutCubic } from "@/core";
import { useCrossfadeSlots, useEasedScalar } from "@/interaction";
import { tokens } from "@/ui/tokens";
import { drawHoverDot } from "../draw/drawHoverDot";
import { drawHoverCluster } from "../draw/drawHoverCluster";

export const HoverLayer = () => {
  const hover = useAtomValue(hoverAtom);
  const run = useAtomValue(stretchedRunAtom);
  const viewport = useAtomValue(viewportAtom);
  // While a version switch is in flight, stretchedRunAtom is the destination
  // run; drawing this layer's saved dot/cluster ids against it would paint
  // a destination overlay during source-side phases. ensureRunAtom already
  // clears hoverAtom at seed, so the internal eased/crossfade hooks drain
  // toward zero during the ~4s transition and are at rest by completion.
  const isTransitioning = useIsTransitioning();

  const displayedDotId = useRef<number | null>(null);
  useLayoutEffect(() => {
    if (hover.dotId !== null) displayedDotId.current = hover.dotId;
  }, [hover.dotId]);

  const dotPulse = useEasedScalar(hover.dotId !== null, tokens.motion.medium, easeOutCubic);

  const clusterSlots = useCrossfadeSlots(
    hover.clusterId,
    tokens.motion.slow,
    easeOutCubic,
  );

  const drawDot = useCallback(
    (g: PixiGraphics) => {
      const id = hover.dotId ?? displayedDotId.current;
      drawHoverDot(g, id, run, viewport, dotPulse);
    },
    [hover.dotId, run, viewport, dotPulse],
  );

  const drawCluster = useCallback(
    (g: PixiGraphics) => {
      drawHoverCluster(g, clusterSlots, run, viewport);
    },
    [clusterSlots, run, viewport],
  );

  if (isTransitioning) return null;
  return (
    <>
      <pixiGraphics draw={drawCluster} zIndex={2} />
      <pixiGraphics draw={drawDot} zIndex={3} />
    </>
  );
};
