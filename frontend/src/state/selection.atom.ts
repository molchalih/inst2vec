import { atom, useAtomValue, useSetAtom } from "jotai";
import { activeRunAtom } from "./run.atom";
import { manifestAtom } from "./manifest.atom";
import { viewportAtom } from "./viewport.atom";
import { parseHash } from "./route.atom";
import type { Manifest } from "@/data";

export type Selection =
  | { kind: "cluster"; clusterId: number }
  | { kind: "creator"; creatorId: number }
  | null;

// Hydrate from the URL hash on init so deep-links (#cluster=7, #user=42)
// survive the first render — the selection→route effect would otherwise
// see a null selection and erase the route's keys before any consumer
// reads them.
const initialSelection = (): Selection => {
  if (typeof globalThis === "undefined") return null;
  const r = parseHash(globalThis.location.hash);
  if (r.user !== undefined) return { kind: "creator", creatorId: r.user };
  if (r.cluster !== undefined) return { kind: "cluster", clusterId: r.cluster };
  return null;
};

export const selectionAtom = atom<Selection>(initialSelection());

export const useSelection = (): Selection => useAtomValue(selectionAtom);

const detailsAvailableForActive = (
  manifest: Manifest | null,
  activeRunId: string | null,
): boolean => {
  if (!manifest || !activeRunId) return false;
  const run = manifest.runs.find((r) => r.id === activeRunId);
  return run?.details_available ?? false;
};

/**
 * Click → selection writer. Action-atom shape so the logic is testable
 * via createStore without React, matching the wheelZoomAtom precedent.
 *
 * Resolves a dot click to either a creator-detail pane (when the
 * active run advertises `details_available` AND the dot's own
 * `has_detail` flag is true) or the cluster pane otherwise.
 */
export const selectDotAtom = atom<null, [number], void>(
  null,
  (get, set, dotId) => {
    const activeRun = get(activeRunAtom);
    if (!activeRun) return;
    const user = activeRun.users.find(([id]) => id === dotId);
    if (!user) return;
    const clusterId = user[3];
    // Noise dots have no meaningful cluster pane; ignore the click.
    if (clusterId < 0) return;

    const hasDetail = user[4];
    const manifest = get(manifestAtom);
    const detailsOn = detailsAvailableForActive(manifest, activeRun.meta.id);
    const next: Selection = detailsOn && hasDetail
      ? { kind: "creator", creatorId: dotId }
      : { kind: "cluster", clusterId };

    // Pin the current viewport into viewportAtom before flipping
    // selection. visibleRectAtom shrinks the instant selection becomes
    // non-null (panel inset), which would otherwise re-derive
    // fittedViewportAtom and snap the camera in the same commit.
    set(viewportAtom, get(viewportAtom));
    set(selectionAtom, next);
  },
);

export const selectClusterAtom = atom<null, [number], void>(
  null,
  (get, set, clusterId) => {
    set(viewportAtom, get(viewportAtom));  // pin
    set(selectionAtom, { kind: "cluster", clusterId });
  },
);

export const useSelectDot = (): ((dotId: number) => void) =>
  useSetAtom(selectDotAtom);

export const useSelectCluster = (): ((clusterId: number) => void) =>
  useSetAtom(selectClusterAtom);

export const useClearSelection = (): (() => void) => {
  const setSelection = useSetAtom(selectionAtom);
  return () => setSelection(null);
};
