import { useCallback, useEffect, useRef } from "react";

/**
 * Reaction-time clock (plan §3 `reaction_time_ms`). Resets whenever `key`
 * changes (a new comparison shown); `read()` returns ms since it was shown.
 */
export function useReactionTimer(key: string | null): () => number | null {
  const shownAt = useRef<number | null>(null);

  useEffect(() => {
    shownAt.current = key === null ? null : performance.now();
  }, [key]);

  return useCallback(() => {
    if (shownAt.current === null) return null;
    return Math.round(performance.now() - shownAt.current);
  }, []);
}
