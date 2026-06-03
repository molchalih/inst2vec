import type { CSSProperties } from "react";
import { type ClipLabelEntry, type ClusterLabel } from "@/data";
import { tokens } from "@/ui/tokens";
import { MAX_TAGS_PER_CATEGORY, formatTag, groundingLabel } from "@/core";
import { Chip, type ChipTone } from "./primitives/Chip";
import { ClusterTagBody } from "./ClusterTagBody";
import { ClusterTagSkeleton } from "./ClusterTagSkeleton";
import { CollapsibleSection } from "./CollapsibleSection";

type Props = {
  index: string;
  /**
   * Per-clip label entries. Omit (along with `cluster`) to render the
   * "not yet available" placeholder.
   */
  clips?: readonly ClipLabelEntry[];
  /**
   * Cluster-level case-agnostic label synthesis. When present, renders the
   * cluster body instead of the per-clip list and drives the section
   * heading via ``cluster.modality``.
   */
  cluster?: ClusterLabel | undefined;
  /**
   * The deferred cluster label (visual / music / multimodal) is still loading —
   * render a tag skeleton in the cluster body's place.
   */
  clusterLoading?: boolean | undefined;
  /**
   * Heading hint used while `cluster` is still loading, so the title reads
   * "Combined" / "Musical" / "Visual" before the label arrives.
   */
  modality?: ClusterLabel["modality"] | undefined;
};

/**
 * Map the case-agnostic ``modality`` string from a ClusterLabel to the
 * section heading shown above the cluster body. Used in one place; kept
 * inline rather than a separate module per Phase E.
 */
function titleFor(modality: ClusterLabel["modality"] | undefined): string {
  switch (modality) {
    case "audio":
      return "Audio";
    case "music":
      return "Musical";
    case "textual":
      return "Textual";
    case "multimodal":
      return "Combined";
    case "visual":
    case undefined:
    default:
      return "Visual";
  }
}

const KIND_ROWS: {
  kind: keyof ClipLabelEntry["tags"];
  tone: ChipTone;
  label: string;
}[] = [
  { kind: "observable", tone: "observable", label: "obs" },
  { kind: "aesthetic",  tone: "aesthetic",  label: "aes" },
  { kind: "community",  tone: "community",  label: "com" },
];

/**
 * Visual labels section. Typography-first card per labelled clip:
 * a small accent-coloured index, a sentence in display weight, and
 * three labelled chip rows (obs / aes / com). For clusters, renders
 * the synthesised cluster identity. For any consumer that omits both
 * `clips` and `cluster`, renders a "not yet available" placeholder.
 *
 * The heading is driven solely by ``cluster.modality``; the per-clip
 * (creator) path always titles "Visual" because those entries are the
 * video clip-labels regardless of the active embedding case.
 */
export const SectionVisual = ({ index, clips, cluster, clusterLoading, modality }: Props) => {
  const title = titleFor(cluster?.modality ?? modality);
  if (cluster !== undefined) {
    return (
      <CollapsibleSection index={index} title={title}>
        <ClusterTagBody cluster={cluster} />
      </CollapsibleSection>
    );
  }
  if (clusterLoading) {
    return (
      <CollapsibleSection index={index} title={title}>
        <ClusterTagSkeleton />
      </CollapsibleSection>
    );
  }
  if (clips === undefined) {
    return (
      <CollapsibleSection index={index} title={title}>
        <p style={todo}>
          Cluster-level visual aggregate not yet available.
        </p>
      </CollapsibleSection>
    );
  }
  if (clips.length === 0) {
    return (
      <CollapsibleSection index={index} title={title}>
        <p style={emptyMsg}>No labelled clips for this creator.</p>
      </CollapsibleSection>
    );
  }
  return (
    <CollapsibleSection index={index} title={title}>
      <ol style={list}>
        {clips.map((c, i) => (
          <ClipRow key={c.clip_id} clip={c} order={letterFor(i)} />
        ))}
      </ol>
    </CollapsibleSection>
  );
};

const ClipRow = ({ clip, order }: { clip: ClipLabelEntry; order: string }) => {
  const warn = clip.validation === "warn";
  return (
    <li style={card(warn)}>
      <div style={cardHeader}>
        <span style={cardIndex}>{order}</span>
        <p style={sentence}>{clip.sentence}</p>
      </div>
      <div style={tagBlock}>
        {KIND_ROWS.map(({ kind, tone, label }) =>
          clip.tags[kind].length === 0 ? null : (
            <div key={kind} style={kindRow}>
              <span style={kindLabel}>{label}</span>
              <div style={chipWrap}>
                {clip.tags[kind].slice(0, MAX_TAGS_PER_CATEGORY).map((entry) => (
                  <Chip
                    key={`${kind}:${entry.tag}`}
                    tone={tone}
                    warning={warn}
                  >
                    <span title={tooltipFor(kind, entry)}>{formatTag(entry.tag)}</span>
                  </Chip>
                ))}
              </div>
            </div>
          ),
        )}
      </div>
    </li>
  );
};

/**
 * Excel-style alphabetic ordinals: 0→"a)", 25→"z)", 26→"aa)", ...
 * Covers every plausible clip count without wrapping.
 */
function letterFor(i: number): string {
  let n = i;
  let s = "";
  do {
    s = String.fromCodePoint(97 + (n % 26)) + s;
    n = Math.floor(n / 26) - 1;
  } while (n >= 0);
  return `${s})`;
}

function tooltipFor(
  kind: keyof ClipLabelEntry["tags"],
  entry: ClipLabelEntry["tags"]["observable"][number] |
         ClipLabelEntry["tags"]["aesthetic"][number],
): string {
  if (kind === "observable") {
    const obs = entry as ClipLabelEntry["tags"]["observable"][number];
    return obs.evidence;
  }
  const g = entry as ClipLabelEntry["tags"]["aesthetic"][number];
  return `${groundingLabel(g.grounded_in)} · confidence ${g.confidence}`;
}

const list: CSSProperties = {
  listStyle: "none",
  padding: 0,
  margin: 0,
  display: "flex",
  flexDirection: "column",
  gap: 10,
  counterReset: "clip",
};

const card = (warn: boolean): CSSProperties => ({
  position: "relative",
  padding: "10px 12px 12px",
  borderRadius: 6,
  // Hairline on the left in accent (or warm if warn), the rest of the
  // border is a faint ink line — gives the card a small editorial
  // gutter without enclosing the content too heavily.
  borderLeft: `2px solid ${warn ? tokens.inspector.tagChip.warningOutline : "var(--accent)"}`,
  borderTop: `1px solid ${tokens.ink.line}`,
  borderRight: `1px solid ${tokens.ink.line}`,
  borderBottom: `1px solid ${tokens.ink.line}`,
  background: "rgb(255 255 255 / 0.015)",
});

const cardHeader: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "22px 1fr",
  gap: 8,
  alignItems: "baseline",
  marginBottom: 10,
};

const cardIndex: CSSProperties = {
  fontFamily: tokens.type.mono,
  fontSize: 10,
  color: "var(--accent)",
  fontVariantNumeric: "tabular-nums",
  letterSpacing: "0.04em",
};

const sentence: CSSProperties = {
  margin: 0,
  fontSize: 12.5,
  lineHeight: 1.5,
  color: tokens.ink.bright,
};

const tagBlock: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 5,
};

const kindRow: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "22px 1fr",
  gap: 8,
  alignItems: "flex-start",
};

const kindLabel: CSSProperties = {
  fontFamily: tokens.type.mono,
  fontSize: 9,
  letterSpacing: "0.10em",
  color: tokens.ink.faint,
  textTransform: "uppercase",
  paddingTop: 4,
};

const chipWrap: CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 5,
};

const todo: CSSProperties = {
  margin: 0,
  fontSize: 12,
  color: tokens.ink.muted,
  fontFamily: tokens.type.mono,
  padding: 10,
  background: "rgb(255 255 255 / 0.03)",
  borderRadius: 6,
};

const emptyMsg: CSSProperties = {
  fontSize: 11,
  color: tokens.ink.faint,
  margin: 0,
  fontStyle: "italic",
};
