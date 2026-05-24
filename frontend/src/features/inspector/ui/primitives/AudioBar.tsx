import { tokens } from "@/ui/tokens";

type AudioBarProps = {
  name: string;
  /** 0..1 */
  value: number;
};

/**
 * Label / bar / numeric value, three-column grid. The fill width is
 * value*100% and the fill colour is the pane's `--accent`.
 */
export const AudioBar = ({ name, value }: AudioBarProps) => {
  const { bar } = tokens.inspector;
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "92px 1fr 30px", gap: 9,
      alignItems: "center", fontSize: 12, padding: "3px 0",
    }}>
      <span style={{ color: tokens.ink.dim }}>{name}</span>
      <div style={{
        height: bar.height,
        background: `rgb(255 255 255 / ${bar.trackAlpha})`,
        borderRadius: bar.radius,
        overflow: "hidden",
      }}>
        <div style={{
          width: `${Math.max(0, Math.min(1, value)) * 100}%`,
          height: "100%",
          background: "var(--accent)",
          borderRadius: bar.radius,
          boxShadow: "0 0 8px color-mix(in srgb, var(--accent) 55%, transparent)",
        }} />
      </div>
      <span style={{
        color: tokens.ink.muted, textAlign: "right",
        fontVariantNumeric: "tabular-nums",
      }}>{value.toFixed(2).slice(1)}</span>
    </div>
  );
};
