import type { CSSProperties } from "react";
import { tokens } from "@/ui/tokens";
import { Skeleton } from "./primitives/Skeleton";

/**
 * Placeholder shown in the tag-hosting section while a cluster's deferred label
 * (tags) loads. Mirrors ``ClusterTagBody``'s two labelled chip rows (rep / aes)
 * so the layout doesn't shift when the real tags arrive.
 */
export const ClusterTagSkeleton = () => (
  <div style={stack}>
    {["rep", "aes"].map((kind) => (
      <div key={kind} style={row}>
        <span style={kindLabel}>{kind}</span>
        <div style={chipWrap}>
          <Skeleton width={68} height={18} />
          <Skeleton width={92} height={18} />
          <Skeleton width={54} height={18} />
        </div>
      </div>
    ))}
  </div>
);

const stack: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 8,
};

const row: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "68px 1fr",
  gap: 8,
  alignItems: "flex-start",
};

const kindLabel: CSSProperties = {
  fontFamily: tokens.type.mono,
  fontSize: 9,
  letterSpacing: "0.10em",
  color: tokens.ink.faint,
  textTransform: "uppercase",
  paddingTop: 4,
};

const chipWrap: CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 5,
};
