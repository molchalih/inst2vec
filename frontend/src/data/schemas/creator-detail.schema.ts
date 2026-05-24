import { z } from "zod";
import { SCHEMA_VERSION } from "./version";
import {
  audioScoresSchema, moodSharesSchema, timbreSharesSchema,
  weightedTagSchema, langShareSchema, distinctivenessEntrySchema,
} from "./cluster-detail.schema";

const speechSchema = z.object({
  detected_share: z.number(),
  top_langs: z.array(langShareSchema),
}).strict();

const captionSchema = z.object({
  top_langs: z.array(langShareSchema),
}).strict();

const postingSchema = z.object({
  median_plays: z.number(),
  median_clip_duration_s: z.number(),
  median_clips_per_week: z.number(),
  engagement_shape_ratio: z.number(),
}).strict();

const nearestOtherClusterSchema = z.object({
  cluster_id: z.number().int(),
  label: z.string(),
  distance: z.number(),
}).strict();

const spatialCreatorSchema = z.object({
  distance_from_centroid: z.number(),
  distance_from_centroid_percentile: z.number(),
  nearest_other_cluster: nearestOtherClusterSchema.nullable(),
}).strict();

export const creatorDetailSchema = z.object({
  version: z.literal(SCHEMA_VERSION),
  user_id: z.number().int().nonnegative(),
  cluster_id: z.number().int(),
  x: z.number(),
  y: z.number(),
  n_clips: z.number().int().nonnegative(),
  audio: audioScoresSchema,
  mood_shares: moodSharesSchema,
  timbre_shares: timbreSharesSchema,
  genre_top: z.array(weightedTagSchema),
  instrument_top: z.array(weightedTagSchema),
  speech: speechSchema,
  caption: captionSchema,
  posting: postingSchema,
  follower_bucket: z.string(),
  activity_span_months: z.number().int(),
  distinctiveness: z.array(distinctivenessEntrySchema),
  spatial: spatialCreatorSchema,
}).strict();

export type CreatorDetail = z.infer<typeof creatorDetailSchema>;
export type NearestOtherCluster = z.infer<typeof nearestOtherClusterSchema>;
