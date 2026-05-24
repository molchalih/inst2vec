import { useLayoutEffect, useRef } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import { stretchedRunAtom, viewportAtom, viewportSizeAtom } from "@/state";

/**
 * First-run fit on initial load, and re-fit on viewport-size change.
 * Case-switch ease is owned by useVersionTransition.
 */
export const useFitOnActiveRun = (): void => {
  const run = useAtomValue(stretchedRunAtom);
  const size = useAtomValue(viewportSizeAtom);
  const setViewport = useSetAtom(viewportAtom);

  const lastRunIdRef = useRef<string | null>(null);
  const lastSizeRef = useRef<{ width: number; height: number }>(size);

  useLayoutEffect(() => {
    if (!run) return;
    const runId = run.meta.id;
    const isFirstRun = lastRunIdRef.current === null;
    const sizeChanged =
      lastSizeRef.current.width !== size.width ||
      lastSizeRef.current.height !== size.height;
    lastRunIdRef.current = runId;
    lastSizeRef.current = size;

    if (isFirstRun || sizeChanged) {
      setViewport(null);
    }
  }, [run, size, setViewport]);
};
