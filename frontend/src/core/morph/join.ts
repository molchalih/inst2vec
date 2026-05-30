import type { CoreAtlasRun, CoreClusterShape } from "./types";

export type JoinedUser = {
  id: number;
  fromXY: [number, number] | null;
  toXY: [number, number] | null;
  fromCluster: number | null;
  toCluster: number | null;
  // HDBSCAN soft membership on each side. 0 when the user is absent on
  // that side; consumers fall back to the present-side value so the
  // aesthetic centrality scale never collapses to noise during fade-in
  // or fade-out.
  fromCentrality: number;
  toCentrality: number;
};

export type JoinedCluster = {
  id: number;
  from: Omit<CoreClusterShape, "id" | "label" | "size" | "has_detail"> | null;
  to: Omit<CoreClusterShape, "id" | "label" | "size" | "has_detail"> | null;
};

type UserRow = { xy: [number, number]; cluster: number; centrality: number };

const userMap = (run: CoreAtlasRun): Map<number, UserRow> => {
  const m = new Map<number, UserRow>();
  for (const [id, x, y, clusterId, , centrality] of run.users) {
    m.set(id, { xy: [x, y], cluster: clusterId, centrality });
  }
  return m;
};

export const joinUsersByCreator = (from: CoreAtlasRun, to: CoreAtlasRun): JoinedUser[] => {
  const fromM = userMap(from);
  const toM = userMap(to);
  const ids = new Set<number>([...fromM.keys(), ...toM.keys()]);
  const out: JoinedUser[] = [];
  for (const id of ids) {
    const f = fromM.get(id) ?? null;
    const t = toM.get(id) ?? null;
    out.push({
      id,
      fromXY: f?.xy ?? null,
      toXY: t?.xy ?? null,
      fromCluster: f?.cluster ?? null,
      toCluster: t?.cluster ?? null,
      fromCentrality: f?.centrality ?? 0,
      toCentrality: t?.centrality ?? 0,
    });
  }
  return out;
};

const clusterShape = (c: CoreClusterShape) =>
  ({ cx: c.cx, cy: c.cy, rx: c.rx, ry: c.ry, angle: c.angle });

export const joinClustersById = (from: CoreAtlasRun, to: CoreAtlasRun): JoinedCluster[] => {
  const fromM = new Map(from.clusters.filter((c) => c.id >= 0).map((c) => [c.id, c]));
  const toM = new Map(to.clusters.filter((c) => c.id >= 0).map((c) => [c.id, c]));
  const ids = new Set<number>([...fromM.keys(), ...toM.keys()]);
  const out: JoinedCluster[] = [];
  for (const id of ids) {
    const f = fromM.get(id) ?? null;
    const t = toM.get(id) ?? null;
    out.push({
      id,
      from: f ? clusterShape(f) : null,
      to: t ? clusterShape(t) : null,
    });
  }
  return out;
};
