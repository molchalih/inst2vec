import { useCallback, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { swipeGesture, type SwipeDir } from "@/core";

interface UseSwipeOptions {
  onSwipe: (direction: SwipeDir) => void;
  /** fired on a release with negligible movement (a tap, not a swipe) */
  onTap?: () => void;
}

interface UseSwipeBindings {
  onPointerDown: (e: ReactPointerEvent) => void;
  onPointerMove: (e: ReactPointerEvent) => void;
  onPointerUp: (e: ReactPointerEvent) => void;
  onPointerCancel: (e: ReactPointerEvent) => void;
}

/**
 * Drag → swipe direction (plan §8.1). Decision math lives in `core/swipeGesture`.
 *
 * Gesture bookkeeping is kept in refs (not state) so the handlers are stable and
 * never miss a fast flick. Pointer capture keeps tracking even if the finger
 * leaves the element. `dragX` is mirrored to state for the horizontal
 * finger-follow; vertical drags are detected on release only.
 */
export function useSwipe({ onSwipe, onTap }: UseSwipeOptions): {
  dragX: number;
  dragging: boolean;
  bind: UseSwipeBindings;
} {
  const [dragX, setDragX] = useState(0);
  const [dragging, setDragging] = useState(false);
  const active = useRef(false);
  const pointerId = useRef<number | null>(null);
  const startRef = useRef({ x: 0, y: 0, width: 1, height: 1 });
  const prev = useRef({ x: 0, y: 0, t: 0 });
  const last = useRef({ x: 0, y: 0, t: 0 });
  const cb = useRef(onSwipe);
  cb.current = onSwipe;
  const tapCb = useRef(onTap);
  tapCb.current = onTap;

  const onPointerDown = useCallback((e: ReactPointerEvent) => {
    if (active.current) return; // ignore secondary touches while one is tracking
    active.current = true;
    pointerId.current = e.pointerId;
    const r = e.currentTarget.getBoundingClientRect();
    startRef.current = { x: e.clientX, y: e.clientY, width: r.width || 1, height: r.height || 1 };
    prev.current = { x: e.clientX, y: e.clientY, t: e.timeStamp };
    last.current = { x: e.clientX, y: e.clientY, t: e.timeStamp };
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* not all targets support capture */
    }
    setDragging(true);
  }, []);

  const onPointerMove = useCallback((e: ReactPointerEvent) => {
    if (!active.current || e.pointerId !== pointerId.current) return;
    // follow the finger only for horizontal-dominant drags
    const dx = e.clientX - startRef.current.x;
    const dy = e.clientY - startRef.current.y;
    setDragX(Math.abs(dx) >= Math.abs(dy) ? dx : 0);
    prev.current = last.current;
    last.current = { x: e.clientX, y: e.clientY, t: e.timeStamp };
  }, []);

  const finish = useCallback((e: ReactPointerEvent) => {
    if (!active.current || e.pointerId !== pointerId.current) return;
    active.current = false;
    pointerId.current = null;
    const s = startRef.current;
    const dt = e.timeStamp - prev.current.t || 1;
    setDragging(false);
    setDragX(0);
    const dx = e.clientX - s.x;
    const dy = e.clientY - s.y;
    // Tap wins first: negligible movement must never be read as a velocity flick.
    if (Math.abs(dx) < 8 && Math.abs(dy) < 8) {
      tapCb.current?.();
      return;
    }
    const dir = swipeGesture({
      dx,
      dy,
      vx: (e.clientX - prev.current.x) / dt,
      vy: (e.clientY - prev.current.y) / dt,
      width: s.width,
      height: s.height,
    });
    if (dir) cb.current(dir);
  }, []);

  const cancel = useCallback((e: ReactPointerEvent) => {
    if (e.pointerId !== pointerId.current) return;
    active.current = false;
    pointerId.current = null;
    setDragging(false);
    setDragX(0);
  }, []);

  return {
    dragX,
    dragging,
    bind: {
      onPointerDown,
      onPointerMove,
      onPointerUp: finish,
      onPointerCancel: cancel,
    },
  };
}
