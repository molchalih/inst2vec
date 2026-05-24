import { atom, useAtomValue } from "jotai";

/**
 * Hover snapshot. `dotId` and `clusterId` are independent: a cursor
 * over a dot inside a cluster sets both, so the cluster highlight
 * stays on while the dot highlight is added on top.
 */
export type HoverState = {
  dotId: number | null;
  clusterId: number | null;
  screenX: number;
  screenY: number;
};

export const hoverAtom = atom<HoverState>({
  dotId: null, clusterId: null, screenX: 0, screenY: 0,
});

export const useHoverState = () => useAtomValue(hoverAtom);
