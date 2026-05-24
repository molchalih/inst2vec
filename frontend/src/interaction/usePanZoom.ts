import { useEffect, type RefObject } from "react";
import { useSetAtom } from "jotai";
import { viewportAtom, wheelZoomAtom } from "@/state";
import { applyDrag, type Transform } from "@/core";

const WHEEL_ZOOM_PER_PIXEL = 0.0015;
// Pointer travel (Manhattan px) past which a release is treated as a pan,
// not a click. The browser still emits a click after a drag; beyond this
// threshold we consume it so useClick's selection handler never runs.
const DRAG_CLICK_THRESHOLD_PX = 4;

/**
 * Attaches pointer (pan) + wheel (zoom anchored on cursor) listeners
 * to the wrapper element. Routes every transition through the pure
 * reducers in core/viewport and writes the result to viewportAtom.
 *
 * Pixi never receives pointer events; this wrapper does.
 */
export const usePanZoom = (ref: RefObject<HTMLElement | null>): void => {
  const setViewport = useSetAtom(viewportAtom);
  const dispatchWheel = useSetAtom(wheelZoomAtom);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let moved = 0;
    let pointerId: number | null = null;
    let suppressClick = false;

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      dragging = true;
      pointerId = e.pointerId;
      lastX = e.clientX;
      lastY = e.clientY;
      moved = 0;
      // Clear any stale suppression from an interaction that ended in
      // pointercancel (which never produces a click to consume it).
      suppressClick = false;
      el.setPointerCapture(e.pointerId);
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!dragging || e.pointerId !== pointerId) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      moved += Math.abs(dx) + Math.abs(dy);
      setViewport((prev) => applyDrag(prev, { x: dx, y: dy }));
    };

    const onPointerUp = (e: PointerEvent) => {
      if (e.pointerId !== pointerId) return;
      dragging = false;
      pointerId = null;
      if (moved > DRAG_CLICK_THRESHOLD_PX) suppressClick = true;
      if (el.hasPointerCapture(e.pointerId)) {
        el.releasePointerCapture(e.pointerId);
      }
    };

    // Capture-phase: runs before useClick's bubble-phase listener on the
    // same wrapper, so consuming the post-drag click here keeps the pan
    // gesture from reaching the selection handler.
    const onClickCapture = (e: MouseEvent) => {
      if (!suppressClick) return;
      suppressClick = false;
      e.stopImmediatePropagation();
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cursor = { x: e.clientX - rect.left, y: e.clientY - rect.top };
      const factor = Math.exp(-e.deltaY * WHEEL_ZOOM_PER_PIXEL);
      dispatchWheel({ cursor, factor });
    };

    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    el.addEventListener("pointercancel", onPointerUp);
    el.addEventListener("click", onClickCapture, { capture: true });
    el.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerUp);
      el.removeEventListener("click", onClickCapture, { capture: true });
      el.removeEventListener("wheel", onWheel);
    };
  }, [ref, setViewport, dispatchWheel]);
};

export type { Transform };
