import type { CSSProperties, ReactNode } from "react";

interface ChipProps {
  children: ReactNode;
  /** dot/border accent colour (e.g. a seed-group hue) */
  hue?: string;
  muted?: boolean;
}

/** A mono micro-pill for keywords, modality tags, and group labels. */
export function Chip({ children, hue, muted = false }: ChipProps) {
  const style: CSSProperties = hue ? { borderColor: hue, color: hue } : {};
  return (
    <span
      className={[
        "inline-flex items-center gap-1.5 rounded-chip border px-chip-px py-chip-py",
        "font-mono text-[10px] uppercase tracking-[0.14em] leading-none",
        muted ? "border-white/10 text-fg-faint" : "border-white/15 text-fg-muted",
      ].join(" ")}
      style={style}
    >
      {hue && (
        <span
          className="h-1.5 w-1.5 rounded-full"
          style={{ backgroundColor: hue }}
          aria-hidden
        />
      )}
      {children}
    </span>
  );
}
