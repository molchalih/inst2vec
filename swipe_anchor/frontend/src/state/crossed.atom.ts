import { atom, useAtom } from "jotai";

/**
 * The creator currently crossed out as the odd one on the active comparison
 * (null = nothing marked yet). Reset whenever the active item advances.
 */
export const crossedAtom = atom<number | null>(null);

export const useCrossed = () => useAtom(crossedAtom);
