import { type ReactNode } from "react";
import { tokens } from "@/ui/tokens";

type ChipProps = {
  children: ReactNode;
  /** 0..1 fill width; omit for an unfilled chip. */
  weight?: number;
};

/**
 * Rounded pill with optional inner-fill bar driven by `weight`. Used
 * for genre/instrument tags (weighted) and distinctiveness tags
 * (unfilled, with an arrow glyph inside).
 */
export const Chip = ({ children, weight }: ChipProps) => {
  const { chip } = tokens.inspector;
  return (
    <span
      style={{
        position: "relative",
        padding: `${chip.paddingY}px ${chip.paddingX}px`,
        border: `1px solid ${tokens.ink.line}`,
        borderRadius: chip.radius,
        fontSize: 11,
        color: tokens.ink.default,
        overflow: "hidden",
        isolation: "isolate",
        whiteSpace: "nowrap",
      }}
    >
      {weight !== undefined && (
        <span
          aria-hidden="true"
          style={{
            position: "absolute", left: 0, top: 0, bottom: 0,
            width: `${Math.max(0, Math.min(1, weight)) * 100}%`,
            background: "var(--accent)",
            opacity: chip.weightFillAlpha,
            zIndex: -1,
          }}
        />
      )}
      {children}
    </span>
  );
};
