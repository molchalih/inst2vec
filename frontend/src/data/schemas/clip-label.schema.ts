import { z } from "zod";

// Backend may emit non-enum values on `validation === "warn"` rows
// (S6 soft warning). Treat confidence as a string here so warn payloads
// remain parseable; trust the backend `validation` flag to gate display.
const confidenceSchema = z.string();

const observableTagSchema = z.object({
  tag: z.string(),
  evidence: z.string(),
}).strict();

const groundedTagSchema = z.object({
  tag: z.string(),
  grounded_in: z.array(z.string()),
  confidence: confidenceSchema,
}).strict();

export const clipLabelEntrySchema = z.object({
  clip_id: z.number().int(),
  shortcode: z.string().nullable(),
  thumbnail_url: z.string().nullable(),
  sentence: z.string(),
  tags: z.object({
    observable: z.array(observableTagSchema),
    aesthetic: z.array(groundedTagSchema),
    community: z.array(groundedTagSchema),
  }).strict(),
  validation: z.enum(["ok", "warn"]),
  warnings: z.array(z.string()),
}).strict();

export type ClipLabelEntry = z.infer<typeof clipLabelEntrySchema>;
export type GroundedTag = z.infer<typeof groundedTagSchema>;
export type ObservableTag = z.infer<typeof observableTagSchema>;
