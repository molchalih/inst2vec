export type TransitionPhase = 0 | 1 | 2 | 3;

// Structural mirrors of AtlasRun / ClusterShape from @/data.
// core/ cannot import from data/; callers pass real AtlasRun objects and
// TS structural typing accepts them — these shapes must stay in sync.
export type CoreClusterShape = {
  id: number;
  label: string;
  cx: number; cy: number;
  rx: number; ry: number;
  angle: number;
  size: number;
  has_detail: boolean;
};

export type CoreAtlasRun = {
  meta: { id: string; case: string; label: string; size: number; details_available: boolean };
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
  users: ReadonlyArray<readonly [number, number, number, number, boolean, number]>;
  clusters: CoreClusterShape[];
};
