import { useEffect, useRef, useState } from "react";
import { sinePulse } from "@/core";

/**
 * Looping rAF hook driving the tracked-dot "breathing" alpha. While `active`,
 * it sets state each frame to `sinePulse(now - start, periodMs, min, max)` and
 * returns it; while inactive it runs no rAF and returns the resting `min`.
 *
 * Unlike useTween (which terminates at t=1), this loops unconditionally while
 * active — tracking is a living state, the one deliberately-continuous
 * animation per the spec. Gating on `active` keeps us from burning a frame
 * loop when nothing is tracked (the caller passes `trackedCreatorId !== null`).
 */
export const useSinePulse = (
  active: boolean,
  periodMs: number,
  min: number,
  max: number,
): number => {
  const [value, setValue] = useState<number>(min);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!active) {
      setValue(min);
      return;
    }
    const start = performance.now();
    const tick = (now: number): void => {
      setValue(sinePulse(now - start, periodMs, min, max));
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [active, periodMs, min, max]);

  return value;
};
