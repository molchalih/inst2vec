import type { CSSProperties } from "react";
import { tokens } from "@/ui/tokens";

type PaneHeaderProps = {
  /** Visible name; cluster pane: cluster.label; creator pane: "user #N". */
  name: string;
  /** Meta line below the name. */
  meta: string;
  /**
   * Optional editorial standfirst rendered below the hairline as the
   * opening line of the body. Cluster pane passes the label summary;
   * creator pane omits it.
   */
  lede?: string | undefined;
};

/**
 * Pane title block: a serif display name over a mono meta line, closed
 * by a hairline, optionally followed by a serif standfirst lede. The top
 * accent atmosphere is the panel-wide wash; this just sits over it.
 */
export const PaneHeader = ({ name, meta, lede }: PaneHeaderProps) => {
  const { header } = tokens.inspector;
  return (
    <section>
      <h3 style={nameStyle(header)}>{name}</h3>
      {meta && <div style={metaStyle(header)}>{meta}</div>}
      <div aria-hidden="true" style={ruleStyle(header)} />
      {lede && <p style={ledeStyle(header)}>{lede}</p>}
    </section>
  );
};

const nameStyle = (h: typeof tokens.inspector.header): CSSProperties => ({
  margin: 0,
  fontFamily: tokens.type.serif,
  fontSize: h.nameSize,
  fontWeight: h.nameWeight,
  fontVariationSettings: `"opsz" ${h.nameOpsz}`,
  lineHeight: 1.1,
  letterSpacing: "0.005em",
  color: tokens.ink.bright,
});

const metaStyle = (h: typeof tokens.inspector.header): CSSProperties => ({
  position: "relative",
  marginTop: 6,
  fontSize: h.metaSize,
  color: tokens.ink.muted,
  fontVariantNumeric: "tabular-nums",
});

const ruleStyle = (h: typeof tokens.inspector.header): CSSProperties => ({
  marginTop: h.ruleGap,
  height: 1,
  background: tokens.ink.line,
});

const ledeStyle = (h: typeof tokens.inspector.header): CSSProperties => ({
  margin: 0,
  marginTop: h.lede.gapTop,
  padding: `${h.lede.paddingY}px ${h.lede.paddingX}px`,
  borderLeft: `${h.lede.barWidth}px solid var(--accent)`,
  borderRadius: h.lede.radius,
  background: h.lede.bg,
  fontFamily: tokens.type.serif,
  fontSize: h.lede.size,
  lineHeight: h.lede.lineHeight,
  letterSpacing: "0.005em",
  color: tokens.ink.dim,
});
