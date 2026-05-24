import { tokens } from "@/ui/tokens";

type NeighborRowProps = {
  label: string;
  color: string;        // cluster colour swatch
  distanceLabel: string; // e.g. "closest", "nearby", "farther"
  onClick: () => void;
};

export const NeighborRow = ({ label, color, distanceLabel, onClick }: NeighborRowProps) => {
  const { neighbor } = tokens.inspector;
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        display: "grid",
        gridTemplateColumns: "10px 1fr auto",
        gap: 9,
        alignItems: "center",
        padding: `${neighbor.paddingY}px ${neighbor.paddingX}px`,
        background: `rgb(255 255 255 / ${neighbor.bgAlpha})`,
        borderRadius: neighbor.radius,
        fontSize: 12, color: tokens.ink.default,
        fontFamily: tokens.type.mono,
        border: "none", textAlign: "left", cursor: "pointer", width: "100%",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = `rgb(255 255 255 / ${neighbor.bgAlphaHover})`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = `rgb(255 255 255 / ${neighbor.bgAlpha})`;
      }}
    >
      <span aria-hidden="true" style={{
        width: 10, height: 10, borderRadius: 999, background: color,
        boxShadow: `0 0 12px ${color}80`,
      }} />
      <span>{label}</span>
      <span style={{ color: tokens.ink.faint, fontSize: 11 }}>{distanceLabel}</span>
    </button>
  );
};
