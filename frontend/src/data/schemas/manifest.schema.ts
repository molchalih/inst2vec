import { z } from "zod";
import { SCHEMA_VERSION } from "./version";

export const embeddingCaseSchema = z.enum([
  "video",
  "sandwich",
  "auditory",
  "spoken",
  "textual",
]);

export const manifestRunSchema = z.object({
  id: z.string(),
  case: embeddingCaseSchema,
  label: z.string(),
  size: z.number().int().nonnegative(),
  details_available: z.boolean(),
});

export const manifestSchema = z.object({
  version: z.literal(SCHEMA_VERSION),
  default_run_id: z.string(),
  runs: z.array(manifestRunSchema).min(1),
});

export type Manifest = z.infer<typeof manifestSchema>;
export type ManifestRun = z.infer<typeof manifestRunSchema>;
export type EmbeddingCase = z.infer<typeof embeddingCaseSchema>;
