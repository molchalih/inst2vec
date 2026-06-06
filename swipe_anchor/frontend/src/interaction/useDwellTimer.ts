import { useCallback, useEffect, useRef } from "react";
import { accumulateDwell, type DwellMap } from "@/core";

/**
 * Per-card dwell tracking (plan §3 `card_dwell_ms`). Accumulates view time per
 * card as the active card changes, and resets when `resetKey` (the comparison
 * id) changes. `read()` returns the dwell map including the in-progress segment.
 */
export function useDwellTimer(
  activeId: number | null,
  resetKey: string | null,
): () => DwellMap {
  const map = useRef<DwellMap>({});
  const prev = useRef<{ id: number | null; t: number; key: string | null }>({
    id: null,
    t: 0,
    key: null,
  });

  useEffect(() => {
    const now = performance.now();
    if (prev.current.key !== resetKey) {
      map.current = {};
    } else if (prev.current.id !== null) {
      map.current = accumulateDwell(map.current, prev.current.id, now - prev.current.t);
    }
    prev.current = { id: activeId, t: now, key: resetKey };
  }, [activeId, resetKey]);

  return useCallback(() => {
    const now = performance.now();
    if (prev.current.id === null) return map.current;
    return accumulateDwell(map.current, prev.current.id, now - prev.current.t);
  }, []);
}
