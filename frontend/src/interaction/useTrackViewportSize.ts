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
    if (typeof globalThis === "undefined") return;

    const sync = () => {
      setSize({ width: globalThis.innerWidth, height: globalThis.innerHeight });
    };

    sync();
    globalThis.addEventListener("resize", sync);
    return () => globalThis.removeEventListener("resize", sync);
  }, [setSize]);
};
