import { useCallback, useEffect, useRef, useState } from "react";

export type CrossfadeSlot = { id: number; pulse: number };

type Internal = CrossfadeSlot & {
  target: 0 | 1;
  startTime: number;
  startValue: number;
};

/**
 * Crossfade between a sequence of singleton ids. When `activeId` changes:
 *   - the previously active slot's target drops to 0 (fading out)
 *   - the new id is added (or reactivated) with target 1 (fading in)
 * Both transitions ease in parallel. Slots whose pulse settles at 0 are
 * dropped. Output preserves arrival order: oldest fading-out first, newest
 * fading-in last, so the caller can draw in that order.
 *
 * One rAF loop ticks all slots; it self-suspends when nothing is moving
 * and self-resumes on the next `activeId` change.
 */
export const useCrossfadeSlots = (
  activeId: number | null,
  duration: number,
  ease: (t: number) => number,
): ReadonlyArray<CrossfadeSlot> => {
  const [slots, setSlots] = useState<Internal[]>([]);
  const slotsRef = useRef<Internal[]>([]);
  const lastActiveRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const durationRef = useRef(duration);
  const easeRef = useRef(ease);
  durationRef.current = duration;
  easeRef.current = ease;

  const tick = useCallback((now: number): void => {
    const prev = slotsRef.current;
    if (prev.length === 0) {
      rafRef.current = null;
      return;
    }
    const d = durationRef.current;
    const e = easeRef.current;
    const advanced = prev.map((s): Internal => {
      const elapsed = now - s.startTime;
      const tn = d <= 0 ? 1 : Math.min(elapsed / d, 1);
      const k = e(tn);
      return { ...s, pulse: s.startValue + (s.target - s.startValue) * k };
    });
    const next = advanced.filter((s) => !(s.target === 0 && s.pulse <= 0));
    slotsRef.current = next;
    setSlots(next);
    const moving = next.some(
      (s) => (s.target === 1 && s.pulse < 1) || (s.target === 0 && s.pulse > 0),
    );
    rafRef.current = moving ? requestAnimationFrame(tick) : null;
  }, []);

  useEffect(() => {
    if (activeId === lastActiveRef.current) return;
    lastActiveRef.current = activeId;
    const now = performance.now();
    const prev = slotsRef.current;
    const demoted: Internal[] = prev.map((s) =>
      s.target === 0 ? s : { ...s, target: 0, startTime: now, startValue: s.pulse },
    );
    let next: Internal[];
    if (activeId === null) {
      next = demoted;
    } else {
      const existing = demoted.find((s) => s.id === activeId);
      if (existing) {
        next = demoted.map((s) =>
          s === existing
            ? { ...s, target: 1, startTime: now, startValue: s.pulse }
            : s,
        );
      } else {
        next = [
          ...demoted,
          { id: activeId, pulse: 0, target: 1, startTime: now, startValue: 0 },
        ];
      }
    }
    slotsRef.current = next;
    setSlots(next);
    if (rafRef.current === null) rafRef.current = requestAnimationFrame(tick);
  }, [activeId, tick]);

  useEffect(
    () => () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    },
    [],
  );

  return slots;
};
