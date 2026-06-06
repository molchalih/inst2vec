/**
 * Single source of truth for visual constants (plan §8 "tokens as the single
 * source of truth"). Consumed by:
 *   1. tailwind.config.ts via CSS variables (see index.html :root)
 *   2. ui/primitives/* and features/* /ui/* via Tailwind classes
 *
 * Aesthetic — "TRIAGE console": a tactile, high-contrast dark judgment tool.
 * Near-black ink, one electric vermilion accent (the cross-out energy) and a
 * cool cyan affirmation, a characterful grotesque display face paired with a
 * field-report mono for labels and numerals.
 */
export const tokens = {
  bg: {
    canvas: "#0b0b0f",
    surface: "#15151c",
    raised: "#1d1d27",
  },
  fg: {
    default: "#f4f1ea",
    muted: "#9a978f",
    faint: "#5f5d58",
  },
  // The two emotional poles of the task: vermilion = "this one is the odd one"
  // (rejection energy), cyan = affirmation / the constellation reward.
  accent: {
    base: "#ff3b1f",
    bright: "#ff6647",
    deep: "#c11f0a",
  },
  affirm: {
    base: "#37e0c4",
    deep: "#129e87",
  },
  // Seed-group hues — a stable colour per confusable family so a card's group
  // reads at a glance. Keyed by the lowercased seed_group string elsewhere.
  group: {
    artist: "#e0a23a",
    music: "#7d8cff",
    fitness: "#54d17a",
    fashion: "#ff7ad0",
    food: "#ff8a3a",
    default: "#9a978f",
  },
  type: {
    display: '"Bricolage Grotesque", "Iowan Old Style", Georgia, serif',
    mono: '"Martian Mono", ui-monospace, "SFMono-Regular", Menlo, monospace',
  },
  line: {
    hair: "rgb(255 255 255 / 0.08)",
    strong: "rgb(255 255 255 / 0.16)",
  },
  card: {
    radius: 22,
    padding: 20,
    gap: 14,
    // Hard offset shadow — the tactile, stacked-paper feel.
    shadow: "0 18px 50px rgb(0 0 0 / 0.55)",
    crossSlashDeg: -18,
  },
  actionBar: {
    height: 92,
    gap: 12,
    buttonRadius: 16,
  },
  chip: {
    paddingX: 10,
    paddingY: 5,
    radius: 999,
    gap: 8,
  },
  pager: {
    dotSize: 7,
    dotGap: 8,
  },
  motion: {
    fast: 130,
    medium: 240,
    slow: 460,
    // Card pager glide — a long, smooth decelerate to centre the swapped-to reel.
    swipe: 560,
    // Snappy, slightly overshooting — tactile button + card feedback.
    easeOut: "cubic-bezier(0.16, 1, 0.3, 1)",
    easePop: "cubic-bezier(0.34, 1.56, 0.64, 1)",
    // Gentle decelerate for the pager glide (smooth, no snap).
    easeGlide: "cubic-bezier(0.22, 1, 0.36, 1)",
  },
  grain: {
    alpha: 0.04,
  },
} as const;

export type Tokens = typeof tokens;

/** Stable accent for a seed group label (case-insensitive); falls back. */
export function groupHue(seedGroup: string | null | undefined): string {
  if (!seedGroup) return tokens.group.default;
  const key = seedGroup.toLowerCase() as keyof typeof tokens.group;
  return tokens.group[key] ?? tokens.group.default;
}
