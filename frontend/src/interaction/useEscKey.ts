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
    globalThis.addEventListener("keydown", onKey);
    return () => globalThis.removeEventListener("keydown", onKey);
  }, [enabled, onEsc]);
};
