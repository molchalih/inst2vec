import { atom, useAtomValue, useSetAtom } from "jotai";

// Point tracking: two orthogonal pieces of domain state.
//   trackingModeAtom   — are clicks arming tracking?
//   trackedCreatorAtom — the pinned creator id (single point; null = none).
// One primitive per concern (no god atom). The toggle and track writers are
// action-atoms (mirroring wheelZoomAtom/selectDotAtom) so they are testable
// headless via createStore.

export const trackingModeAtom = atom<boolean>(false);
export const trackedCreatorAtom = atom<number | null>(null);

/**
 * Flip tracking mode. Turning it OFF clears the tracked creator (off means
 * off — ring/pulse removed, pills re-enabled, rAF stops).
 */
export const toggleTrackingAtom = atom<null, [], void>(null, (get, set) => {
  const next = !get(trackingModeAtom);
  set(trackingModeAtom, next);
  if (!next) set(trackedCreatorAtom, null);
});

/**
 * Raw setter for the tracked creator (replace semantics — single point).
 * Guards (noise / tracking-mode) live in resolveDotClickAtom, not here, so
 * this stays trivially testable.
 */
export const trackCreatorAtom = atom<null, [number], void>(
  null,
  (_get, set, creatorId) => {
    set(trackedCreatorAtom, creatorId);
  },
);

export const useTrackingMode = (): boolean => useAtomValue(trackingModeAtom);
export const useTrackedCreator = (): number | null =>
  useAtomValue(trackedCreatorAtom);
export const useToggleTracking = (): (() => void) =>
  useSetAtom(toggleTrackingAtom);
