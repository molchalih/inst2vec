import { atom, useAtomValue } from "jotai";

export type ViewportSize = { width: number; height: number };

const initial: ViewportSize =
  typeof window !== "undefined"
    ? { width: window.innerWidth, height: window.innerHeight }
    : { width: 0, height: 0 };

export const viewportSizeAtom = atom<ViewportSize>(initial);

export const useViewportSize = () => useAtomValue(viewportSizeAtom);
