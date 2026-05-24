import { type ReactNode } from "react";
import { tokens } from "@/ui/tokens";

type PaneShellProps = { onClose: () => void; children: ReactNode };

/**
 * Inner chrome: scrollable column with section padding + a floating
 * close × in the top-right. Renders inside the Panel slide-in.
 */
export const PaneShell = ({ onClose, children }: PaneShellProps) => {
  const { panel } = tokens;
  return (
    <div style={{ position: "relative", flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
      <button
        type="button"
        aria-label="Close panel"
        onClick={onClose}
        style={{
          position: "absolute",
          top: panel.close.top, right: panel.close.right,
          background: "none", border: "none",
          color: tokens.ink.faint, fontSize: panel.close.size, lineHeight: 1,
          cursor: "pointer", padding: "4px 8px", zIndex: 2,
        }}
      >×</button>
      <div style={{
        flex: 1, minHeight: 0, overflowY: "auto",
        // No left padding: PaneBody owns the left margin so it lines up
        // with the panel edge rather than being inset by scroll padding.
        padding: `${panel.paddingY}px ${panel.paddingX}px ${panel.paddingY}px 0`,
        display: "flex", flexDirection: "column",
      }}>
        {children}
      </div>
    </div>
  );
};
