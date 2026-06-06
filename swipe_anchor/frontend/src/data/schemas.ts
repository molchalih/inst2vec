import { z } from "zod";

/**
 * Wire schemas for the backend contract (`swipe_anchor.backend.app`). Zod is the
 * single validation boundary: every response is parsed before it reaches state/.
 * When the backend contract evolves, bump CONTRACT_VERSION and migrate here —
 * never loosen a schema silently.
 */
export const CONTRACT_VERSION = 1;

export const clipMediaSchema = z.object({
  clip_id: z.number().int(),
  video_url: z.string().nullable().default(null),
  poster_url: z.string().nullable().default(null),
});
export type ClipMedia = z.infer<typeof clipMediaSchema>;

export const creatorCardSchema = z.object({
  creator_id: z.number().int(),
  seed_group: z.string().nullable().default(null),
  rep_clip_ids: z.array(z.number().int()).default([]),
  caption_keywords: z.record(z.string(), z.unknown()).default({}),
  audio_summary: z.record(z.string(), z.unknown()).default({}),
  clips: z.array(clipMediaSchema).default([]),
});
export type CreatorCard = z.infer<typeof creatorCardSchema>;

export const batchItemSchema = z.object({
  assignment_id: z.string(),
  comparison_id: z.string(),
  seed_group: z.string().nullable().default(null),
  expected_modality: z.string().nullable().default(null),
  creators: z.array(creatorCardSchema).length(3),
});
export type BatchItem = z.infer<typeof batchItemSchema>;

export const nextBatchResponseSchema = z.object({
  items: z.array(batchItemSchema),
});
export type NextBatchResponse = z.infer<typeof nextBatchResponseSchema>;

export const respondResponseSchema = z.object({
  accepted: z.boolean(),
  n_triplets: z.number().int(),
  retired: z.boolean(),
});
export type RespondResponse = z.infer<typeof respondResponseSchema>;

export interface RespondPayload {
  assignment_id: string;
  odd_creator_id: number | null;
  confidence: number;
  reaction_time_ms: number | null;
  card_dwell_ms: Record<number, number> | null;
  /** the reel actually shown per creator at judge time: {creator_id: clip_id} */
  shown_clips: Record<number, number> | null;
  expanded: boolean;
}
