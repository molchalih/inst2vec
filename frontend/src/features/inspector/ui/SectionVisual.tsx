import type { CSSProperties } from "react";
import { type ClipLabelEntry, type ClusterLabel } from "@/data";
import { tokens } from "@/ui/tokens";
import { clusterWarningLabel, groundingLabel, warningLabel } from "@/core";
import { Chip, type ChipTone } from "./primitives/Chip";
import { CollapsibleSection } from "./CollapsibleSection";

type Props = {
  index: string;
  /**
   * Per-clip label entries. Omit (along with `cluster`) to render a TODO
   * placeholder.
   */
  clips?: readonly ClipLabelEntry[];
  /**
   * Cluster-level case-agnostic label synthesis. When present, renders the
   * cluster body instead of the per-clip list and drives the section
   * heading via ``cluster.modality``.
   */
  cluster?: ClusterLabel;
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
 * `clips` and `cluster`, renders a TODO placeholder.
 *
 * The heading is driven solely by ``cluster.modality``; the per-clip
 * (creator) path always titles "Visual" because those entries are the
 * video clip-labels regardless of the active embedding case.
 */
export const SectionVisual = ({ index, clips, cluster }: Props) => {
  const title = titleFor(cluster?.modality);
  if (cluster !== undefined) {
    return (
      <CollapsibleSection index={index} title={title}>
        <ClusterBody cluster={cluster} />
      </CollapsibleSection>
    );
  }
  if (clips === undefined) {
    return (
      <CollapsibleSection index={index} title={title}>
        <p style={todo}>
          TODO: cluster-level visual aggregate (not yet produced).
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

// ---------------------------------------------------------------------------
// Cluster body
// ---------------------------------------------------------------------------

const ClusterBody = ({ cluster }: { cluster: ClusterLabel }) => {
  const warn = cluster.validation === "warn";
  return (
    <div style={clusterStack}>
      {/* Summary now lives in the pane header standfirst (PaneHeader lede). */}

      {/* Dominant repertoire (case-flattened) */}
      {cluster.repertoire.length > 0 && (
        <div style={clusterKindRow}>
          <span style={kindLabel}>rep</span>
          <div style={chipWrap}>
            {cluster.repertoire.map((e) => (
              <Chip key={e.tag} tone="observable" warning={warn}>
                <span title={`${e.recurrence} · ${e.description}`}>{e.tag}</span>
              </Chip>
            ))}
          </div>
        </div>
      )}

      {/* Dominant aesthetic logic */}
      {cluster.aesthetic_logic.length > 0 && (
        <div style={clusterKindRow}>
          <span style={kindLabel}>aes</span>
          <div style={chipWrap}>
            {cluster.aesthetic_logic.map((e) => (
              <Chip key={e.tag} tone="aesthetic" warning={warn}>
                <span title={`${groundingLabel(e.grounded_in)} · ${e.description}`}>{e.tag}</span>
              </Chip>
            ))}
          </div>
        </div>
      )}

      {/* Taste signalling */}
      <CautiousBlock label="taste" block={cluster.taste_signalling} warn={warn} />

      {/* Visibility orientation */}
      <CautiousBlock label="visibility" block={cluster.visibility_orientation} warn={warn} />

      {/* Internal variations */}
      {cluster.internal_variations.length > 0 && (
        <div style={clusterKindRow}>
          <span style={kindLabel}>variations</span>
          <ul style={variationList}>
            {cluster.internal_variations.map((v) => (
              <li key={v.variation} style={variationItem}>
                <span style={variationName}>{v.variation}</span>
                <span style={variationDesc}>{v.description}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Boundary notes */}
      {cluster.boundary_notes && (
        <div style={paragraphBlock}>
          <span style={kindLabel}>boundary</span>
          <p style={paragraphText}>{cluster.boundary_notes}</p>
        </div>
      )}

      {/* Tool tags */}
      {cluster.tool_tags.length > 0 && (
        <div style={clusterKindRow}>
          <span style={kindLabel}>tags</span>
          <p style={toolTagsRow}>
            {cluster.tool_tags.map((t) => (
              <span key={t} style={toolTagsText}>{t}</span>
            ))}
          </p>
        </div>
      )}

      {/* Warn line */}
      {warn && cluster.warnings.length > 0 && (
        <p style={warnLine}>
          ⚠ {cluster.warnings.map(clusterWarningLabel).join(" · ")}
        </p>
      )}
    </div>
  );
};

const CautiousBlock = ({
  label,
  block,
  warn,
}: {
  label: string;
  block: { label: string; description: string; confidence: string };
  warn: boolean;
}) => (
  <div style={clusterKindRow}>
    <span style={kindLabel}>{label}</span>
    <div style={cautiousBody}>
      <div style={cautiousHead}>
        <span style={cautiousName(warn)}>{block.label}</span>
        <span style={cautiousConfidence}>{block.confidence}</span>
      </div>
      <p style={paragraphText}>{block.description}</p>
    </div>
  </div>
);

// ---------------------------------------------------------------------------
// Clip body (unchanged)
// ---------------------------------------------------------------------------

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
                {clip.tags[kind].map((entry) => (
                  <Chip
                    key={`${kind}:${entry.tag}`}
                    tone={tone}
                    warning={warn}
                  >
                    <span title={tooltipFor(kind, entry)}>{entry.tag}</span>
                  </Chip>
                ))}
              </div>
            </div>
          ),
        )}
      </div>
      {clip.warnings.length > 0 && (
        <p style={warnLine}>
          ⚠ {clip.warnings.map(warningLabel).join(" · ")}
        </p>
      )}
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

// ---------------------------------------------------------------------------
// Shared styles
// ---------------------------------------------------------------------------

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

const clusterKindRow: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "68px 1fr",
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

const warnLine: CSSProperties = {
  margin: "10px 0 0",
  fontSize: 10,
  color: tokens.inspector.tagChip.warningOutline,
  fontFamily: tokens.type.mono,
  letterSpacing: "0.02em",
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

// ---------------------------------------------------------------------------
// Cluster-specific styles
// ---------------------------------------------------------------------------

const clusterStack: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 8,
};

const paragraphBlock: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "68px 1fr",
  gap: 8,
  alignItems: "flex-start",
};

const paragraphText: CSSProperties = {
  margin: 0,
  fontSize: 12,
  lineHeight: 1.5,
  color: tokens.ink.default,
};

const variationList: CSSProperties = {
  listStyle: "none",
  padding: 0,
  margin: 0,
  display: "flex",
  flexDirection: "column",
  gap: 4,
};

const variationItem: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 1,
};

const variationName: CSSProperties = {
  fontFamily: tokens.type.mono,
  fontSize: 10,
  color: tokens.ink.bright,
  letterSpacing: "0.03em",
};

const variationDesc: CSSProperties = {
  fontSize: 11,
  color: tokens.ink.muted,
  lineHeight: 1.4,
};

// cautiousRow reuses clusterKindRow geometry (defined above)

const cautiousBody: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 3,
};

const cautiousHead: CSSProperties = {
  display: "flex",
  alignItems: "baseline",
  gap: 6,
};

const cautiousName = (warn: boolean): CSSProperties => ({
  fontSize: 12,
  fontFamily: tokens.type.mono,
  color: warn ? tokens.inspector.tagChip.warningOutline : tokens.ink.bright,
  letterSpacing: "0.02em",
});

const cautiousConfidence: CSSProperties = {
  fontSize: 9,
  fontFamily: tokens.type.mono,
  color: tokens.ink.faint,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
};

const toolTagsRow: CSSProperties = {
  margin: 0,
  display: "flex",
  flexWrap: "wrap",
  gap: 5,
};

const toolTagsText: CSSProperties = {
  fontFamily: tokens.type.mono,
  fontSize: 10,
  color: tokens.ink.muted,
  letterSpacing: "0.03em",
};
