import { atom } from "jotai";
import { BruteForceHitTest, type HitTest } from "@/core";
import { stretchedRunAtom } from "./stretched-run.atom";
import { transitionAtom } from "./transition.atom";
import { introAtom } from "./intro.atom";

/**
 * Null while a transition or intro animation is in flight — useHover
 * bails on null so hover/click are inert during the entrance, and the
 * tooltip + dot overlay drain naturally via the existing crossfade.
 */
export const hitTestAtom = atom<HitTest | null>((get) => {
  if (get(transitionAtom) || get(introAtom)) return null;
  const run = get(stretchedRunAtom);
  if (!run) return null;
  const dots = run.users.map(([id, x, y, clusterId]) => ({ id, x, y, clusterId }));
  const ellipses = run.clusters
    .filter((c) => c.id >= 0)
    .map((c) => ({ id: c.id, cx: c.cx, cy: c.cy, rx: c.rx, ry: c.ry, angle: c.angle }));
  return new BruteForceHitTest(dots, ellipses);
});
