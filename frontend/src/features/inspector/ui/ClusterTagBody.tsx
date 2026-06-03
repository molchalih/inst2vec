import type { CSSProperties } from "react";
import type { ClusterLabel } from "@/data";
import { tokens } from "@/ui/tokens";
import { MAX_TAGS_PER_CATEGORY, formatTag, groundingLabel } from "@/core";
import { Chip } from "./primitives/Chip";

type Props = { cluster: ClusterLabel };

/**
 * Case-agnostic descriptive-tag body for a cluster label block. Renders
 * the dominant repertoire / aesthetic-logic chips (each capped at
 * ``MAX_TAGS_PER_CATEGORY``), the taste-signalling and
 * visibility-orientation cautious blocks, internal variations, and
 * boundary notes.
 *
 * Extracted from ``SectionVisual`` so every modality section that
 * carries a cluster label (Visual, Spoken, Textual) renders the same
 * markup, tokens, and formatting helpers — the section heading is owned
 * by the enclosing ``CollapsibleSection``, not by this body.
 */
export const ClusterTagBody = ({ cluster }: Props) => {
  const warn = cluster.validation === "warn";
  return (
    <div style={clusterStack}>
      {/* case-flattened across modalities */}
      {cluster.repertoire.length > 0 && (
        <div style={clusterKindRow}>
          <span style={kindLabel}>rep</span>
          <div style={chipWrap}>
            {cluster.repertoire.slice(0, MAX_TAGS_PER_CATEGORY).map((e) => (
              <Chip key={e.tag} tone="observable" warning={warn}>
                <span title={`${e.recurrence} · ${e.description}`}>{formatTag(e.tag)}</span>
              </Chip>
            ))}
          </div>
        </div>
      )}

      {cluster.aesthetic_logic.length > 0 && (
        <div style={clusterKindRow}>
          <span style={kindLabel}>aes</span>
          <div style={chipWrap}>
            {cluster.aesthetic_logic.slice(0, MAX_TAGS_PER_CATEGORY).map((e) => (
              <Chip key={e.tag} tone="aesthetic" warning={warn}>
                <span title={`${groundingLabel(e.grounded_in)} · ${e.description}`}>{formatTag(e.tag)}</span>
              </Chip>
            ))}
          </div>
        </div>
      )}

      <CautiousBlock label="taste" block={cluster.taste_signalling} warn={warn} />

      <CautiousBlock label="visibility" block={cluster.visibility_orientation} warn={warn} />

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

      {cluster.boundary_notes && (
        <div style={paragraphBlock}>
          <span style={kindLabel}>boundary</span>
          <p style={paragraphText}>{cluster.boundary_notes}</p>
        </div>
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

const clusterStack: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 8,
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

