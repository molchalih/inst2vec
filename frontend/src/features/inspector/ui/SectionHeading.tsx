import type { CSSProperties, ReactNode } from "react";
import { tokens } from "@/ui/tokens";

type SectionHeadingProps = {
  /** Two-digit catalogue index, e.g. "01". Rendered in the accent colour. */
  index: string;
  children: ReactNode;
};

/**
 * Indexed, hairline-ruled section head shared by every inspector
 * section ("01 — SOUND"). The leading index picks up the live
 * `--accent` cluster colour; sections past the first carry a top rule
 * so the pane reads as a sequence of catalogue entries.
 */
export const SectionHeading = ({ index, children }: SectionHeadingProps) => {
  const { sectionHead } = tokens.inspector;
  const { ink } = tokens;
  const ruled = index !== "01";
  return (
    <h4
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: sectionHead.indexGap,
        margin: 0,
        paddingTop: sectionHead.gapTop,
        marginBottom: sectionHead.gapBottom,
        borderTop: ruled ? `1px solid ${ink.line}` : "none",
        fontFamily: tokens.type.mono,
        fontSize: sectionHead.size,
        fontWeight: 500,
        letterSpacing: sectionHead.tracking,
        textTransform: "uppercase",
        color: ink.faint,
      }}
    >
      <span style={{ color: "var(--accent)", fontVariantNumeric: "tabular-nums" }}>{index}</span>
      <span aria-hidden="true" style={dash}>—</span>
      <span>{children}</span>
    </h4>
  );
};

const dash: CSSProperties = { color: "var(--accent)", opacity: 0.5 };
