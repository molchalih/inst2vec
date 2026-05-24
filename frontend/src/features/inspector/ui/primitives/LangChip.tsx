import { tokens } from "@/ui/tokens";

type LangChipProps = {
  code: string;   // ISO 639-1
  /** 0..1 share within the column. */
  share: number;
};

export const LangChip = ({ code, share }: LangChipProps) => {
  const { langChip } = tokens.inspector;
  return (
    <span style={{
      padding: `${langChip.paddingY}px ${langChip.paddingX}px`,
      borderRadius: langChip.radius,
      background: `rgb(255 255 255 / ${langChip.bgAlpha})`,
      fontSize: 11, color: tokens.ink.dim,
      fontVariantNumeric: "tabular-nums",
      display: "inline-flex", gap: 4,
    }}>
      <span style={{ color: tokens.ink.default, fontWeight: 500 }}>{code.toUpperCase()}</span>
      <span>{Math.round(share * 100)}%</span>
    </span>
  );
};
