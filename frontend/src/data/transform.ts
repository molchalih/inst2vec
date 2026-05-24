import { stretchEllipse } from "@/core";
import type { AtlasRun } from "./types";

/**
 * Returns a new AtlasRun with users and clusters anisotropically
 * stretched so the run's raw UMAP bounds exactly fill the given
 * viewport, centered on the origin. The interactive transform layer
 * (pan/zoom) keeps uniform scale, so cluster shapes only ever distort
 * once at load — never during interaction.
 *
 * Returns the input run untouched when the viewport is degenerate.
 */
export const stretchRun = (
  run: AtlasRun,
  viewportWidth: number,
  viewportHeight: number,
): AtlasRun => {
  if (viewportWidth <= 0 || viewportHeight <= 0) return run;

  const { minX, maxX, minY, maxY } = run.bounds;
  const rawWidth = Math.max(maxX - minX, 1e-6); // 1e-6 guards against divide-by-zero on degenerate bounds
  const rawHeight = Math.max(maxY - minY, 1e-6);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const sx = viewportWidth / rawWidth;
  const sy = viewportHeight / rawHeight;

  const users: AtlasRun["users"] = run.users.map(([id, x, y, clusterId, hasDetail]) => [
    id,
    (x - cx) * sx,
    (y - cy) * sy,
    clusterId,
    hasDetail,
  ]);

  const clusters = run.clusters.map((c) => {
    const stretched = stretchEllipse(
      { cx: c.cx - cx, cy: c.cy - cy, rx: c.rx, ry: c.ry, angle: c.angle },
      sx,
      sy,
    );
    return { ...c, ...stretched };
  });

  const halfW = viewportWidth / 2;
  const halfH = viewportHeight / 2;

  return {
    meta: run.meta,
    bounds: { minX: -halfW, maxX: halfW, minY: -halfH, maxY: halfH },
    users,
    clusters,
  };
};
