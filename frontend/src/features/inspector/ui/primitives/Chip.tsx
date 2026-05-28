import { type ReactNode } from "react";
import { tokens } from "@/ui/tokens";

export type ChipTone = "observable" | "aesthetic" | "community";

type ChipProps = {
  children: ReactNode;
  /** 0..1 fill width; omit for an unfilled chip. */
  weight?: number;
  /** Selects bg/fg/border from `inspector.tagChip[tone]`. */
  tone?: ChipTone;
  /** Marks the chip as belonging to a warn-validation clip. */
  warning?: boolean;
};

/**
 * Rounded pill with optional inner-fill bar driven by `weight`. Used
 * for genre/instrument tags (weighted) and distinctiveness tags
 * (unfilled, with an arrow glyph inside). `tone` opt-in picks one of
 * the clip-label tag-kind palettes; `warning` swaps the border for
 * the warn-outline colour.
 */
export const Chip = ({ children, weight, tone, warning }: ChipProps) => {
  const { chip, tagChip } = tokens.inspector;
  const palette = tone ? tagChip[tone] : null;
  return (
    <span
      style={{
        position: "relative",
        padding: `${chip.paddingY}px ${chip.paddingX}px`,
        border: `1px solid ${
          warning ? tagChip.warningOutline : (palette?.border ?? tokens.ink.line)
        }`,
        background: palette?.bg ?? "transparent",
        borderRadius: chip.radius,
        fontSize: 11,
        color: palette?.fg ?? tokens.ink.default,
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
