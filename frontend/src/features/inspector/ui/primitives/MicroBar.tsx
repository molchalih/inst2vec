import { tokens } from "@/ui/tokens";

type MicroBarProps = {
  name: string;
  /** 0..1 */
  value: number;
  /** CSS color for the bar fill. */
  color: string;
};

/**
 * Used in two side-by-side columns (mood + timbre) inside the
 * Character section. Smaller and tighter than AudioBar.
 */
export const MicroBar = ({ name, value, color }: MicroBarProps) => {
  const { microBar, bar } = tokens.inspector;
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: `${microBar.labelColWidth}px 1fr ${microBar.valueColWidth}px`,
      gap: microBar.rowGap, alignItems: "center", fontSize: 11, padding: "2px 0",
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
          height: "100%", background: color, borderRadius: bar.radius,
        }} />
      </div>
      <span style={{
        textAlign: "right", color: tokens.ink.muted,
        fontVariantNumeric: "tabular-nums",
      }}>{value.toFixed(2).slice(1)}</span>
    </div>
  );
};
