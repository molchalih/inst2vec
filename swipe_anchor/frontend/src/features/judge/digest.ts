import type { CreatorCard } from "@/data";

/**
 * Defensive readers for the digest payload. Digests are bare until the
 * pipeline→app export lands (Phase 0), and the jsonb shapes are loose, so these
 * tolerate missing/partial data and always return display-ready arrays.
 */
export function captionKeywords(card: CreatorCard): string[] {
  const raw = card.caption_keywords;
  const fromKeywords = raw["keywords"];
  if (Array.isArray(fromKeywords)) {
    return fromKeywords.filter((k): k is string => typeof k === "string").slice(0, 8);
  }
  return Object.keys(raw).slice(0, 8);
}

export function audioTerms(card: CreatorCard): string[] {
  const raw = card.audio_summary;
  const out: string[] = [];
  for (const key of ["genre_labels", "moodtheme_labels", "instrument_labels"]) {
    const v = raw[key];
    if (Array.isArray(v)) {
      out.push(...v.filter((t): t is string => typeof t === "string"));
    }
  }
  return out.slice(0, 6);
}
