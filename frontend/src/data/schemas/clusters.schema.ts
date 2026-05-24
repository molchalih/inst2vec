import { z } from "zod";
import { SCHEMA_VERSION } from "./version";

export const clusterSchema = z.object({
  id: z.number().int(),
  label: z.string(),
  cx: z.number(),
  cy: z.number(),
  rx: z.number().nonnegative(),
  ry: z.number().nonnegative(),
  angle: z.number(),
  size: z.number().int().nonnegative(),
  has_detail: z.boolean(),
});

export const clustersFileSchema = z.object({
  version: z.literal(SCHEMA_VERSION),
  run_id: z.string(),
  clusters: z.array(clusterSchema),
});

export type ClustersFile = z.infer<typeof clustersFileSchema>;
export type ClusterShape = z.infer<typeof clusterSchema>;
