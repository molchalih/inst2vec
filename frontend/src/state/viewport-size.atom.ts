import { atom, useAtomValue } from "jotai";

export type ViewportSize = { width: number; height: number };

const initial: ViewportSize =
  typeof globalThis === "undefined"
    ? { width: 0, height: 0 }
    : { width: globalThis.innerWidth, height: globalThis.innerHeight };

export const viewportSizeAtom = atom<ViewportSize>(initial);

export const useViewportSize = () => useAtomValue(viewportSizeAtom);
