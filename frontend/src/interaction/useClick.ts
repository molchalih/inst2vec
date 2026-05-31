import { useEffect, type RefObject } from "react";
import { useAtomValue } from "jotai";
import { hoverAtom, useResolveDotClick, useSelectCluster, useClearSelection } from "@/state";

/**
 * Wrapper-DOM click → selection writer. Priority: hovered dot → resolve the
 * dot click (track that creator in tracking mode, else select it). Else a
 * hovered cluster region (inside an ellipse, no dot) → select that cluster.
 * Else empty canvas → clear selection.
 */
export const useClick = (ref: RefObject<HTMLElement | null>): void => {
  const hover = useAtomValue(hoverAtom);
  const resolveDot = useResolveDotClick();
  const selectCluster = useSelectCluster();
  const clear = useClearSelection();

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onClick = (): void => {
      if (hover.dotId != null) resolveDot(hover.dotId);
      // Ellipses are drawn only for real clusters; the >= 0 guard is
      // defensive since noise has no pane and selectClusterAtom is unguarded.
      else if (hover.clusterId != null && hover.clusterId >= 0) selectCluster(hover.clusterId);
      else clear();
    };
    el.addEventListener("click", onClick);
    return () => el.removeEventListener("click", onClick);
  }, [ref, hover.dotId, hover.clusterId, resolveDot, selectCluster, clear]);
};
