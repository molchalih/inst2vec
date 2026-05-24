import { useEffect, type RefObject } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  hitTestAtom, hoverAtom, viewportAtom,
} from "@/state";
import { screenToWorld } from "@/core";
import { tokens } from "@/ui/tokens";

/**
 * Mousemove → screenToWorld(viewport) → hitTest → write hoverAtom.
 * Throttled to one requestAnimationFrame; pointerleave clears.
 *
 * Pixi never sees pointers — this hook owns input from the wrapper div.
 * The hover hit radius is screen-pixel sized (tokens.hover.dotRadiusPx)
 * so the affordance feels constant under zoom; converted to world space
 * at call time via the current viewport scale.
 */
export const useHover = (ref: RefObject<HTMLElement | null>): void => {
  const setHover = useSetAtom(hoverAtom);
  const hitTest = useAtomValue(hitTestAtom);
  const viewport = useAtomValue(viewportAtom);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    let rafId: number | null = null;
    let lastClientX = 0;
    let lastClientY = 0;

    const flush = (): void => {
      rafId = null;
      if (!hitTest) {
        setHover({ dotId: null, clusterId: null, screenX: lastClientX, screenY: lastClientY });
        return;
      }
      const rect = el.getBoundingClientRect();
      const world = screenToWorld(
        { x: lastClientX - rect.left, y: lastClientY - rect.top },
        viewport,
      );
      const hoverRadiusWorld = tokens.hover.dotRadiusPx / Math.max(viewport.scale, 1e-6);
      const dotHit = hitTest.nearestDot(world, hoverRadiusWorld);
      const dotId = dotHit ? dotHit.id : null;
      const dotCluster = dotHit && dotHit.clusterId >= 0 ? dotHit.clusterId : null;
      const clusterId = dotCluster ?? hitTest.ellipseAt(world);
      setHover({ dotId, clusterId, screenX: lastClientX, screenY: lastClientY });
    };

    const onPointerMove = (e: PointerEvent): void => {
      lastClientX = e.clientX;
      lastClientY = e.clientY;
      if (rafId !== null) return;
      rafId = requestAnimationFrame(flush);
    };

    const onPointerLeave = (): void => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      setHover({ dotId: null, clusterId: null, screenX: 0, screenY: 0 });
    };

    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerleave", onPointerLeave);
    return () => {
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerleave", onPointerLeave);
      if (rafId !== null) cancelAnimationFrame(rafId);
    };
  }, [ref, setHover, hitTest, viewport]);
};
