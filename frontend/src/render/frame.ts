import {
  colorForCluster, introStagger,
  centralityRadiusScale,
  type Vec2, type IntroPhase,
} from "@/core";
import type { AtlasRun } from "@/data";
import { tokens } from "@/ui/tokens";

export type DrawableUser = {
  id: number;
  x: number;
  y: number;
  color: string;
  alpha: number;
  // Per-user radius multiplier on top of frame.radiusScale. 1 outside the
  // wave-pulse window; set by the scale-pop schedule during phase 2.
  radiusScale: number;
};

export type DrawableEllipse = {
  id: number;
  cx: number; cy: number;
  rx: number; ry: number;
  angle: number;
  color: string;
};

export type DotsFrame = {
  users: ReadonlyArray<DrawableUser>;
  alphaScale: number;
  radiusScale: number;
};

export type EllipsesFrame = {
  ellipses: ReadonlyArray<DrawableEllipse>;
  alphaScale: number;
  strokeWidthScale: number;
};

export const runToDotsFrame = (run: AtlasRun | null): DotsFrame => {
  if (!run) return { users: [], alphaScale: 1, radiusScale: 1 };
  const users: DrawableUser[] = run.users.map(([id, x, y, clusterId, _hd, centrality]) => ({
    id, x, y,
    color: colorForCluster(clusterId, tokens.palette.cluster, tokens.palette.noise),
    alpha: clusterId < 0 ? tokens.noise.alpha : tokens.dot.alpha,
    radiusScale: centralityRadiusScale(clusterId, centrality, tokens.dot.centrality),
  }));
  return { users, alphaScale: 1, radiusScale: 1 };
};

export const runToEllipsesFrame = (run: AtlasRun | null): EllipsesFrame => {
  if (!run) return { ellipses: [], alphaScale: 1, strokeWidthScale: 1 };
  const ellipses: DrawableEllipse[] = run.clusters
    .filter((c) => c.id >= 0)
    .map((c) => ({
      id: c.id, cx: c.cx, cy: c.cy, rx: c.rx, ry: c.ry, angle: c.angle,
      color: colorForCluster(c.id, tokens.palette.cluster, tokens.palette.noise),
    }));
  return { ellipses, alphaScale: 1, strokeWidthScale: 1 };
};

export const runToIntroDotsFrame = (
  run: AtlasRun | null,
  centerWorld: Vec2,
  phase: IntroPhase,
  progress: number,
): DotsFrame => {
  if (!run) return { users: [], alphaScale: 1, radiusScale: 1 };
  const { maxStaggerFrac } = tokens.motion.intro;
  // Fade phase: dots pinned at center (flightProgress 0). Settle: landed.
  const flightProgressForPhase1 = phase === 1 ? progress : 1;
  const flightProgress = phase < 1 ? 0 : flightProgressForPhase1;
  const users: DrawableUser[] = run.users.map(([id, x, y, clusterId, _hd, centrality]) => {
    const p = introStagger(flightProgress, id, maxStaggerFrac);
    return {
      id,
      x: centerWorld.x + (x - centerWorld.x) * p,
      y: centerWorld.y + (y - centerWorld.y) * p,
      color: colorForCluster(clusterId, tokens.palette.cluster, tokens.palette.noise),
      alpha: clusterId < 0 ? tokens.noise.alpha : tokens.dot.alpha,
      radiusScale: centralityRadiusScale(clusterId, centrality, tokens.dot.centrality),
    };
  });
  // Fade the whole stacked blob as ONE group, not per dot. During the fade
  // phase every dot overlaps at the centre; per-dot alpha accumulates across
  // the ~hundreds of stacked circles and saturates to opaque almost
  // immediately, so a per-dot ramp is invisible. `alphaScale` is the group-
  // opacity signal; drawDots realizes it as a post-composite AlphaFilter (a
  // plain layer `g.alpha` would fold back into per-fragment alpha and saturate
  // just the same). Full opacity once the dots start flying out (phase >= 1).
  const alphaScale = phase === 0 ? progress : 1;
  return { users, alphaScale, radiusScale: 1 };
};
