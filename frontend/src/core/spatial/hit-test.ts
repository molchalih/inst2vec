import { isPointInEllipse, type Ellipse } from "../geom/ellipse";
import type { Vec2 } from "../geom/vec2";

export type Dot = { id: number; x: number; y: number; clusterId: number };
export type LabeledEllipse = Ellipse & { id: number };

export interface HitTest {
  nearestDot(world: Vec2, radius: number): { id: number; clusterId: number } | null;
  ellipseAt(world: Vec2): number | null;
}

export class BruteForceHitTest implements HitTest {
  constructor(
    private readonly dots: ReadonlyArray<Dot>,
    private readonly ellipses: ReadonlyArray<LabeledEllipse>,
  ) {}

  nearestDot(world: Vec2, radius: number): { id: number; clusterId: number } | null {
    let bestId: number | null = null;
    let bestClusterId = 0;
    let bestDist2 = radius * radius;
    // Epsilon tolerance lets later-added dots win FP ties (stack-on-top semantics).
    const eps = 1e-9;
    for (const d of this.dots) {
      const dx = d.x - world.x;
      const dy = d.y - world.y;
      const dist2 = dx * dx + dy * dy;
      if (dist2 <= bestDist2 + eps) {
        bestDist2 = dist2;
        bestId = d.id;
        bestClusterId = d.clusterId;
      }
    }
    return bestId === null ? null : { id: bestId, clusterId: bestClusterId };
  }

  ellipseAt(world: Vec2): number | null {
    let bestId: number | null = null;
    let bestArea = Infinity;
    for (const e of this.ellipses) {
      if (!isPointInEllipse(world, e)) continue;
      const area = e.rx * e.ry;
      if (area < bestArea) {
        bestArea = area;
        bestId = e.id;
      }
    }
    return bestId;
  }
}
