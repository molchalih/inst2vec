import { useEffect } from "react";

/**
 * Calls `onEsc` when the user presses Escape while `enabled` is true.
 * Listener is attached to `window`; nothing else changes about focus
 * or event propagation — modal owners decide what to do on close.
 */
export const useEscKey = (enabled: boolean, onEsc: () => void): void => {
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === "Escape") onEsc();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled, onEsc]);
};
