import { atom, useAtomValue } from "jotai";
import { stretchRun } from "@/data";
import type { AtlasRun } from "@/data";
import { activeRunAtom } from "./run.atom";
import { viewportSizeAtom } from "./viewport-size.atom";

/**
 * The active run after a one-time anisotropic stretch into the
 * viewport. Layers, hit-tests, and fit-bounds consume this; the raw
 * activeRunAtom stays available for anything that needs UMAP-space
 * coordinates (currently nothing). Recomputes whenever the active
 * run or the viewport size changes; otherwise is referentially
 * stable.
 *
 * Returns null when there is no active run or when the viewport is
 * still degenerate (e.g. on the very first render in non-browser
 * environments).
 */
export const stretchedRunAtom = atom<AtlasRun | null>((get) => {
  const run = get(activeRunAtom);
  if (!run) return null;
  const { width, height } = get(viewportSizeAtom);
  if (width <= 0 || height <= 0) return null;
  return stretchRun(run, width, height);
});

export const useStretchedRun = () => useAtomValue(stretchedRunAtom);
