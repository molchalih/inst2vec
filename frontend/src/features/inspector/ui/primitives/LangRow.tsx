import { tokens } from "@/ui/tokens";

type LangRowProps = {
  /** ISO 639-1 code, e.g. "en". */
  code: string;
  /** 0..1 share within the column. */
  share: number;
};

/**
 * Row layout for language shares: monospace ISO code on the left, a
 * full-width accent bar in the middle, and a tabular-nums percentage
 * on the right. Used in both `SectionSpoken` and `SectionTextual`.
 */
export const LangRow = ({ code, share }: LangRowProps) => {
  const { bar } = tokens.inspector;
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "30px 1fr 32px",
      gap: 9,
      alignItems: "center",
      fontSize: 11,
      padding: "2px 0",
    }}>
      <span style={{
        color: tokens.ink.default,
        fontFamily: tokens.type.mono,
        letterSpacing: "0.04em",
        fontWeight: 500,
      }}>
        {code.toUpperCase()}
      </span>
      <div style={{
        height: bar.height,
        background: `rgb(255 255 255 / ${bar.trackAlpha})`,
        borderRadius: bar.radius,
        overflow: "hidden",
      }}>
        <div style={{
          width: `${Math.max(0, Math.min(1, share)) * 100}%`,
          height: "100%",
          background: "var(--accent)",
          borderRadius: bar.radius,
          boxShadow: "0 0 8px color-mix(in srgb, var(--accent) 55%, transparent)",
        }} />
      </div>
      <span style={{
        textAlign: "right",
        color: tokens.ink.muted,
        fontVariantNumeric: "tabular-nums",
      }}>
        {Math.round(share * 100)}%
      </span>
    </div>
  );
};
