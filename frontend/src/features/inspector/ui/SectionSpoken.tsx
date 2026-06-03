import type { CSSProperties } from "react";
import { tokens } from "@/ui/tokens";
import type { ClusterLabel, LangShare } from "@/data";
import { ClusterTagBody } from "./ClusterTagBody";
import { ClusterTagSkeleton } from "./ClusterTagSkeleton";
import { LangRow } from "./primitives/LangRow";
import { Skeleton } from "./primitives/Skeleton";
import { CollapsibleSection } from "./CollapsibleSection";

type Loaded = { detected_share: number; top_langs: LangShare[] };
type Props = {
  index: string;
  loaded?: Loaded;
  /**
   * Cluster-level descriptive label block for the spoken (audio) case.
   * Present only on the spoken-case run; renders the dominant
   * repertoire / aesthetic tags beneath the language distribution,
   * mirroring the Visual section's cluster body. Narrowed to the audio
   * modality: `ClusterPane` only routes the spoken-case label block
   * here, so mis-routing another modality is a compile error rather
   * than a silent UI mismatch.
   */
  label?: (ClusterLabel & { modality: "audio" }) | undefined;
  /**
   * The deferred label for this (audio) cluster is still loading — show a tag
   * skeleton in its place. Mutually exclusive with `label` in practice.
   */
  labelLoading?: boolean;
};

/**
 * Speech section: a headline metric ("speech detected in N%") and a
 * stack of per-language bar rows, each carrying the ISO code, a share
 * bar, and a percentage. When the active run is the spoken case, a
 * `label` block follows with the cluster's descriptive tags.
 */
export const SectionSpoken = ({ index, loaded, label, labelLoading }: Props) => (
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
        {label ? (
          <>
            <Subhead>Tags</Subhead>
            <ClusterTagBody cluster={label} />
          </>
        ) : labelLoading ? (
          <>
            <Subhead>Tags</Subhead>
            <ClusterTagSkeleton />
          </>
        ) : null}
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

const Subhead = ({ children }: { children: string }) => (
  <div style={subhead}>{children}</div>
);

const subhead: CSSProperties = {
  fontSize: 9,
  letterSpacing: "0.18em",
  textTransform: "uppercase",
  color: tokens.ink.faint,
  fontFamily: tokens.type.mono,
  marginTop: 14,
  marginBottom: 6,
};

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
