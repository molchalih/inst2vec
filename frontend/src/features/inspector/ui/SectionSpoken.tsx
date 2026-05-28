import type { CSSProperties } from "react";
import { tokens } from "@/ui/tokens";
import type { LangShare } from "@/data";
import { LangRow } from "./primitives/LangRow";
import { Skeleton } from "./primitives/Skeleton";
import { CollapsibleSection } from "./CollapsibleSection";

type Loaded = { detected_share: number; top_langs: LangShare[] };
type Props = { index: string; loaded?: Loaded };

/**
 * Speech section: a headline metric ("speech detected in N%") and a
 * stack of per-language bar rows, each carrying the ISO code, a share
 * bar, and a percentage.
 */
export const SectionSpoken = ({ index, loaded }: Props) => (
  <CollapsibleSection index={index} title="Spoken">
    {loaded ? (
      <>
        <div style={headline}>
          <span style={headlineNumber}>
            {Math.round(loaded.detected_share * 100)}%
          </span>
          <span style={headlineMeta}>of clips contain speech</span>
        </div>
        <div style={listGap}>
          {loaded.top_langs.length === 0 ? (
            <p style={emptyMsg}>No language detected.</p>
          ) : (
            loaded.top_langs.map((l) => (
              <LangRow key={l.code} code={l.code} share={l.share} />
            ))
          )}
        </div>
      </>
    ) : (
      <>
        <Skeleton height={18} width={160} />
        <div style={{ height: 12 }} />
        <Skeleton height={10} />
        <div style={{ height: 6 }} />
        <Skeleton height={10} />
      </>
    )}
  </CollapsibleSection>
);

const headline: CSSProperties = {
  display: "flex",
  alignItems: "baseline",
  gap: 8,
  marginBottom: 12,
};
const headlineNumber: CSSProperties = {
  fontSize: 22,
  fontFamily: tokens.type.mono,
  color: "var(--accent)",
  fontVariantNumeric: "tabular-nums",
  fontWeight: 500,
};
const headlineMeta: CSSProperties = {
  fontSize: 11,
  color: tokens.ink.muted,
  letterSpacing: "0.02em",
};
const listGap: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 5,
};
const emptyMsg: CSSProperties = {
  fontSize: 11,
  color: tokens.ink.faint,
  margin: 0,
  fontStyle: "italic",
};
