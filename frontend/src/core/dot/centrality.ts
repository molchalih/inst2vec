export type CentralityScaleParams = {
  min: number;
  max: number;
  // Convex (>1) compresses the high end of the [0,1] centrality range,
  // so most users (HDBSCAN soft membership clusters near 1) no longer
  // collapse to a single near-max radius. Noise points (clusterId < 0)
  // bypass the curve and stay at scale 1.
  gamma: number;
};

export const centralityRadiusScale = (
  clusterId: number,
  centrality: number,
  params: CentralityScaleParams,
): number => {
  if (clusterId < 0) return 1;
  const c = Math.min(1, Math.max(0, centrality));
  const curved = Math.pow(c, params.gamma);
  return params.min + (params.max - params.min) * curved;
};
