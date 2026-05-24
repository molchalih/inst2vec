import { useEffect, useRef, useState } from "react";

const defaultEquals = <T>(a: T, b: T): boolean => Object.is(a, b);

/**
 * Generic requestAnimationFrame tween over a scalar. Returns the current
 * eased value; when `target` changes, eases from the current value toward
 * the new target over `duration` ms using `ease`. An in-flight tween is
 * cancelled on every new target.
 *
 * Phase 2 only needs scalar tweens (pulse 0→1). Structural targets, if
 * ever needed, compose multiple useTween calls in the consumer.
 *
 * Equality defaults to `Object.is`. Pass a custom `equals` to debounce
 * effectively-equal targets (e.g. floats that differ only in the last
 * bit due to re-renders).
 */
export const useTween = (
  target: number,
  duration: number,
  ease: (t: number) => number,
  equals: (a: number, b: number) => boolean = defaultEquals,
): number => {
  const [value, setValue] = useState<number>(target);
  const rafRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(0);
  const startValueRef = useRef<number>(target);
  const targetRef = useRef<number>(target);

  useEffect(() => {
    if (equals(targetRef.current, target) && rafRef.current === null) {
      return;
    }
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    startValueRef.current = value;
    targetRef.current = target;
    startTimeRef.current = performance.now();

    const tick = (now: number): void => {
      const elapsed = now - startTimeRef.current;
      const t = duration <= 0 ? 1 : Math.min(elapsed / duration, 1);
      const k = ease(t);
      const v = startValueRef.current + (target - startValueRef.current) * k;
      setValue(v);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        rafRef.current = null;
      }
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
    // `value` is read intentionally via closure capture at tween start,
    // not as a dep — that would restart on every frame.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, duration, ease, equals]);

  return value;
};
