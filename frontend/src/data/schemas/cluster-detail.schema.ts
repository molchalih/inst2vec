import { z } from "zod";
import { SCHEMA_VERSION } from "./version";

export const audioScoresSchema = z.object({
  approachability: z.number(),
  engagement: z.number(),
  danceability: z.number(),
}).strict();

export const moodSharesSchema = z.object({
  happy: z.number(),
  sad: z.number(),
  relaxed: z.number(),
  aggressive: z.number(),
  party: z.number(),
}).strict();

export const timbreSharesSchema = z.object({
  acoustic: z.number(),
  electronic: z.number(),
  instrumental: z.number(),
  female_voice: z.number(),
  bright: z.number(),
  tonal: z.number(),
}).strict();

export const weightedTagSchema = z.object({
  label: z.string(),
  weight: z.number().nonnegative(),
}).strict();

export const langShareSchema = z.object({
  code: z.string(),
  share: z.number(),
}).strict();

export const distinctivenessEntrySchema = z.object({
  feature: z.string(),
  cohort_value: z.number(),
  baseline_mean: z.number(),
  baseline_std: z.number(),
  z: z.number(),
}).strict();

const ellipseSchema = z.object({
  cx: z.number(),
  cy: z.number(),
  rx: z.number().nonnegative(),
  ry: z.number().nonnegative(),
  angle: z.number(),
}).strict();

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

const nearestClusterSchema = z.object({
  cluster_id: z.number().int(),
  label: z.string(),
  distance: z.number(),
}).strict();

const spatialClusterSchema = z.object({
  compactness: z.number(),
  nearest_clusters: z.array(nearestClusterSchema),
}).strict();

export const clusterDetailSchema = z.object({
  version: z.literal(SCHEMA_VERSION),
  cluster_id: z.number().int(),
  label: z.string(),
  size: z.number().int().nonnegative(),
  ellipse: ellipseSchema,
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
  spatial: spatialClusterSchema,
}).strict();

export type ClusterDetail = z.infer<typeof clusterDetailSchema>;
export type AudioScores = z.infer<typeof audioScoresSchema>;
export type MoodShares = z.infer<typeof moodSharesSchema>;
export type TimbreShares = z.infer<typeof timbreSharesSchema>;
export type WeightedTag = z.infer<typeof weightedTagSchema>;
export type LangShare = z.infer<typeof langShareSchema>;
export type DistinctivenessEntry = z.infer<typeof distinctivenessEntrySchema>;
export type NearestCluster = z.infer<typeof nearestClusterSchema>;
