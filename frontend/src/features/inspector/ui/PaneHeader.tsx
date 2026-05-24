import type { CSSProperties } from "react";
import { tokens } from "@/ui/tokens";

type PaneHeaderProps = {
  /** Visible name; cluster pane: cluster.label; creator pane: "user #N". */
  name: string;
  /** Meta line below the name. */
  meta: string;
};

/**
 * Pane title block: a serif display name over a mono meta line, closed
 * by a hairline. The top accent atmosphere is the panel-wide wash; this
 * just sits over it.
 */
export const PaneHeader = ({ name, meta }: PaneHeaderProps) => {
  const { header } = tokens.inspector;
  return (
    <section>
      <h3 style={nameStyle(header)}>{name}</h3>
      {meta && <div style={metaStyle(header)}>{meta}</div>}
      <div aria-hidden="true" style={ruleStyle(header)} />
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
