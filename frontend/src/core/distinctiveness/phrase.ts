// DistinctivenessEntry mirrors data/schemas/cluster-detail.schema.ts.
// Defined locally because core/ cannot import from data/ (layer rule).
export type DistinctivenessEntry = {
  feature: string;
  cohort_value: number;
  baseline_mean: number;
  baseline_std: number;
  z: number;
};

const TABLE: Record<string, { pos: string; neg: string }> = {
  is_electronic:            { pos: "electronic",    neg: "non-electronic" },
  is_acoustic:              { pos: "acoustic",      neg: "non-acoustic" },
  is_instrumental:          { pos: "instrumental",  neg: "non-instrumental" },
  is_happy:                 { pos: "happy",         neg: "not happy" },
  is_sad:                   { pos: "sad",           neg: "not sad" },
  is_relaxed:               { pos: "relaxed",       neg: "not relaxed" },
  is_aggressive:            { pos: "aggressive",    neg: "not aggressive" },
  is_party:                 { pos: "party",         neg: "not party" },
  is_bright:                { pos: "bright",        neg: "not bright" },
  is_tonal:                 { pos: "tonal",         neg: "not tonal" },
  is_female_voice:          { pos: "female voice",  neg: "not female voice" },
  danceability:             { pos: "danceable",     neg: "undanceable" },
  engagement:               { pos: "engaging",      neg: "low engagement" },
  approachability:          { pos: "approachable",  neg: "challenging" },
  median_clip_duration_s:   { pos: "longer clips",  neg: "shorter clips" },
  median_clips_per_week:    { pos: "posts often",   neg: "posts rarely" },
  engagement_shape_ratio:   { pos: "viral shape",   neg: "even shape" },
  activity_span_months:     { pos: "long history",  neg: "short history" },
};

export type Phrase = { label: string; arrow: "↑" | "↓" };

export const phraseFor = (e: DistinctivenessEntry): Phrase => {
  const positive = e.z >= 0;
  const row = TABLE[e.feature];
  let label: string;
  if (row) {
    label = positive ? row.pos : row.neg;
  } else {
    label = e.feature;
  }
  return { label, arrow: positive ? "↑" : "↓" };
};
