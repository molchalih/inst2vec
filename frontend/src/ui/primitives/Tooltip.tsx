import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

type TooltipProps = {
  x: number;
  y: number;
  visible: boolean;
  children: ReactNode;
};

const OFFSET = 12;

/**
 * Positioned tooltip clamped to the viewport: flips horizontally when
 * the right edge would exceed innerWidth, vertically when the bottom
 * edge would exceed innerHeight. Stays presentational; consumers pass
 * raw cursor coordinates.
 */
export const Tooltip = ({ x, y, visible, children }: TooltipProps) => {
  const ref = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState({ left: x + OFFSET, top: y - OFFSET });

  useLayoutEffect(() => {
    if (!visible) return;
    const el = ref.current;
    if (!el) return;
    const { width, height } = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const flipX = x + OFFSET + width > vw;
    const flipY = y - OFFSET + height > vh;
    const rawLeft = flipX ? x - OFFSET - width : x + OFFSET;
    const rawTop = flipY ? y - OFFSET - height : y - OFFSET;
    // Clamp to viewport so a cursor near any edge (including the top,
    // where the non-flipped branch can go negative) stays on-screen.
    const left = Math.max(0, Math.min(rawLeft, vw - width));
    const top = Math.max(0, Math.min(rawTop, vh - height));
    setPos({ left, top });
  }, [x, y, visible]);

  if (!visible) return null;
  return (
    <div
      ref={ref}
      style={{ left: pos.left, top: pos.top }}
      className={[
        "pointer-events-none fixed z-50",
        "rounded-tooltip px-tooltip-px py-tooltip-py text-xs text-fg-default",
        "bg-bg-canvas/35 backdrop-blur-glass",
        "border border-fg-default/10 shadow-glass",
      ].join(" ")}
    >
      {children}
    </div>
  );
};
