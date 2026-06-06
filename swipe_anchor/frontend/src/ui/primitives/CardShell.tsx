import type { ReactNode } from "react";
import { tokens } from "../tokens";

interface CardShellProps {
  children: ReactNode;
  crossed?: boolean;
}

/**
 * The chunky creator-card container. When `crossed`, a decisive vermilion slash
 * stamps across it and the content desaturates — the "this is the odd one" beat.
 */
export function CardShell({ children, crossed = false }: CardShellProps) {
  return (
    <article className="relative h-full w-full overflow-hidden rounded-card border border-white/10 bg-bg-raised shadow-card">
      <div
        className="h-full w-full transition duration-medium ease-out"
        style={crossed ? { filter: "grayscale(0.85) brightness(0.7)" } : undefined}
      >
        {children}
      </div>

      {crossed && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div
            className="absolute left-[-10%] right-[-10%] h-[5px] bg-accent shadow-[0_0_24px_rgb(255_59_31/0.8)]"
            style={{
              transform: `rotate(${tokens.card.crossSlashDeg}deg)`,
              animation: `triage-stamp ${tokens.motion.fast}ms ${tokens.motion.easePop} both`,
              ["--slash-deg" as string]: `${tokens.card.crossSlashDeg}deg`,
            }}
          />
          <span
            className="rounded-chip border border-accent bg-bg-canvas/80 px-3 py-1 font-mono text-[11px] uppercase tracking-[0.2em] text-accent"
            style={{ transform: `rotate(${tokens.card.crossSlashDeg}deg)` }}
          >
            odd one
          </span>
        </div>
      )}
    </article>
  );
}
