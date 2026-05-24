import { useEffect } from "react";
import { useSetAtom } from "jotai";
import { viewportSizeAtom } from "@/state";

/**
 * Single source of window-size truth for the app. Writes
 * viewportSizeAtom on mount and on every resize. No other hook or
 * atom listens to window resize directly.
 */
export const useTrackViewportSize = (): void => {
  const setSize = useSetAtom(viewportSizeAtom);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const sync = () => {
      setSize({ width: window.innerWidth, height: window.innerHeight });
    };

    sync();
    window.addEventListener("resize", sync);
    return () => window.removeEventListener("resize", sync);
  }, [setSize]);
};
