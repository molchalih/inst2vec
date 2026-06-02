export const TAG_KIND_ORDER = ["observable", "aesthetic", "community"] as const;
export type TagKind = (typeof TAG_KIND_ORDER)[number];

const KNOWN_WARNINGS: Record<string, string> = {
  tag_count_out_of_range: "tag count out of range",
  tag_length_out_of_range: "tag length out of range",
  duplicate_tag_within_kind: "duplicate tag within kind",
  hashtag_like_tag: "hashtag-like tag",
  invalid_confidence: "invalid confidence",
  ungrounded_tag_reference: "ungrounded tag reference",
  sentence_length_out_of_range: "sentence length out of range",
};

export function groundingLabel(grounded: readonly string[]): string {
  if (grounded.length === 0) return "";
  return `grounded in: ${grounded.join(", ")}`;
}

export function warningLabel(code: string): string {
  return KNOWN_WARNINGS[code] ?? code;
}

/**
 * Display a label tag as a readable phrase. The clip-tagger occasionally emits
 * snake_cased tags (e.g. ``playful_intellectualism``); show underscores as
 * spaces. (The durable fix is upstream in the labelling stage; this keeps the
 * UI clean meanwhile.)
 */
export function formatTag(tag: string): string {
  return tag.replaceAll("_", " ");
}
