import { useEffect, useRef, useState, type ReactNode } from "react";
import { tokens } from "@/ui/tokens";

type PanelProps = {
  open: boolean;
  side?: "left" | "right";
  onClose: () => void;
  children: ReactNode;
  /**
   * Slide animation duration in ms. Defaults to `tokens.motion.medium`
   * for the generic chromeless slide-in; consumers that own the rest
   * of an open/close orchestration (e.g. the inspector) pass their own
   * duration so the slide stays in sync with downstream cascades.
   */
  durationMs?: number;
};

const offClass = (side: "left" | "right"): string =>
  side === "right" ? "translate-x-full" : "-translate-x-full";

/**
 * Chromeless slide-in. No header, no footer — those belong to the
 * consumer. The panel just owns the slide animation, Esc handling,
 * focus trap, and the background card.
 */
export const Panel = ({
  open,
  side = "left",
  onClose,
  children,
  durationMs,
}: PanelProps) => {
  const slideMs = durationMs ?? tokens.motion.medium;
  const ref = useRef<HTMLDivElement | null>(null);
  const [mounted, setMounted] = useState(open);
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (open) {
      setMounted(true);
      // Two frames, not one: the first lets the browser paint the
      // off-screen start state, the second flips to on-screen so the
      // slide-in has a real starting frame. A single rAF can collapse
      // both into one paint (the surrounding re-renders on open make
      // this likely), and the panel appears without animating.
      let inner = 0;
      const outer = requestAnimationFrame(() => {
        inner = requestAnimationFrame(() => setShow(true));
      });
      return () => {
        cancelAnimationFrame(outer);
        cancelAnimationFrame(inner);
      };
    }
    setShow(false);
    const t = window.setTimeout(() => setMounted(false), slideMs);
    return () => clearTimeout(t);
  }, [open, slideMs]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (!open) return;
    const el = ref.current;
    if (!el) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key !== "Tab") return;
      const focusables = el.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (focusables.length === 0) return;
      const first = focusables[0]!;
      const last = focusables[focusables.length - 1]!;
      const active = document.activeElement as HTMLElement | null;
      if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };
    el.addEventListener("keydown", onKey);
    return () => el.removeEventListener("keydown", onKey);
  }, [open]);

  if (!mounted) return null;

  const sideClass = side === "right" ? "right-0" : "left-0";
  const translate = show ? "translate-x-0" : offClass(side);

  return (
    <aside
      ref={ref}
      role="dialog"
      aria-modal="false"
      className={`fixed top-0 ${sideClass} h-full text-fg-default flex flex-col transition-transform ${translate}`}
      style={{
        width: `${tokens.panel.widthPx}px`,
        background: tokens.panel.bg,
        backdropFilter: `blur(${tokens.glass.blurPx}px)`,
        boxShadow: tokens.glass.shadow,
        transitionDuration: `${slideMs}ms`,
      }}
    >
      {children}
    </aside>
  );
};
