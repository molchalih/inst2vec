import { atom, useAtomValue } from "jotai";
import { tokens } from "@/ui/tokens";
import type { Rect } from "@/core";
import { selectionAtom } from "./selection.atom";
import { viewportSizeAtom } from "./viewport-size.atom";

/**
 * Canvas sub-rect that is *not* behind chrome (today: the inspector panel).
 * Drives camera-focus targets and fit math. When selection is null the
 * rect equals the full viewport; the panel is the only chrome inset
 * for v1 (spec §13 extension point).
 */
export const visibleRectAtom = atom<Rect>((get) => {
  const { width, height } = get(viewportSizeAtom);
  const sel = get(selectionAtom);
  if (!sel) return { x: 0, y: 0, width, height };
  const w = tokens.panel.widthPx;
  return { x: w, y: 0, width: Math.max(0, width - w), height };
});

export const useVisibleRect = (): Rect => useAtomValue(visibleRectAtom);
