export const TAG_KIND_ORDER = ["observable", "aesthetic", "community"] as const;
export type TagKind = (typeof TAG_KIND_ORDER)[number];

/**
 * Upper bound on how many tags a single descriptive category renders in the
 * inspector — one cap shared by the per-clip tag rows (obs/aes/com) and the
 * cluster body's dominant repertoire / aesthetic-logic rows, so no category
 * in any modality shows more than this.
 */
export const MAX_TAGS_PER_CATEGORY = 5;

export function groundingLabel(grounded: readonly string[]): string {
  if (grounded.length === 0) return "";
  return `grounded in: ${grounded.join(", ")}`;
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
