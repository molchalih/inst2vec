import type { CSSProperties, ReactNode } from "react";
import { tokens } from "@/ui/tokens";

type PaneBodyProps = {
  /**
   * When true the column owns the panel's full height; children stack
   * top-down with a fixed `panel.sectionGap` between them. The column
   * never distributes leftover space across sections, and never adds a
   * scrollbar of its own.
   */
  fill?: boolean;
  children: ReactNode;
};

/**
 * Pane content frame: sets the monospace catalogue voice (the serif
 * name overrides it) and owns the left margin. The accent atmosphere is
 * the panel-wide wash behind it (inherited `--accent`); this draws no
 * chrome of its own.
 */
export const PaneBody = ({ fill = false, children }: PaneBodyProps) => (
  <div
    style={{
      flex: 1,
      minHeight: 0,
      display: "flex",
      flexDirection: "column",
      paddingLeft: tokens.panel.paddingX,
      fontFamily: tokens.type.mono,
      color: tokens.ink.default,
    }}
  >
    <div style={{
      flex: 1,
      minHeight: 0,
      display: "flex",
      flexDirection: "column",
      gap: tokens.panel.sectionGap,
    } as CSSProperties}>
      {fill ? <>{children}</> : children}
    </div>
  </div>
);
