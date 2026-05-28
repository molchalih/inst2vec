import type { CSSProperties } from "react";
import { formatCompact } from "@/core";
import { tokens } from "@/ui/tokens";
import { Skeleton } from "./primitives/Skeleton";
import { CollapsibleSection } from "./CollapsibleSection";

type Loaded = {
  follower_bucket: string;
  posting: {
    median_plays: number;
    median_clip_duration_s: number;
    median_clips_per_week: number;
    engagement_shape_ratio: number;
  };
};

type Props = { index: string; loaded?: Loaded };

/** Follower bucket + posting cadence key-value rows. */
export const SectionAudience = ({ index, loaded }: Props) => (
  <CollapsibleSection index={index} title="Audience">
    {loaded ? (
      <dl style={kvGrid}>
        <Row label="followers" value={loaded.follower_bucket} />
        <Row label="median plays" value={formatCompact(loaded.posting.median_plays)} />
        <Row label="clip length" value={`${loaded.posting.median_clip_duration_s.toFixed(1)}s`} />
        <Row label="clips / week" value={loaded.posting.median_clips_per_week.toFixed(1)} />
        <Row label="engagement shape" value={`${loaded.posting.engagement_shape_ratio.toFixed(1)}×`} />
      </dl>
    ) : (
      <>
        {[0, 1, 2, 3, 4].map((i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 12, marginBottom: 6 }}>
            <Skeleton height={10} />
            <Skeleton height={10} width={40} />
          </div>
        ))}
      </>
    )}
  </CollapsibleSection>
);

const Row = ({ label, value }: { label: string; value: string }) => (
  <>
    <dt style={{ color: tokens.ink.muted }}>{label}</dt>
    <dd style={{ color: tokens.ink.default, margin: 0, fontVariantNumeric: "tabular-nums" }}>{value}</dd>
  </>
);

const kvGrid: CSSProperties = {
  display: "grid", gridTemplateColumns: "1fr auto",
  rowGap: 5, columnGap: 12, fontSize: 12,
};
