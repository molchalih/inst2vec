import type { CSSProperties } from "react";
import { formatCompact } from "@/core";
import { tokens } from "@/ui/tokens";
import type { LangShare } from "@/data";
import { LangChip } from "./primitives/LangChip";
import { Skeleton } from "./primitives/Skeleton";
import { SectionHeading } from "./SectionHeading";

type Loaded = {
  follower_bucket: string;
  posting: {
    median_plays: number;
    median_clip_duration_s: number;
    median_clips_per_week: number;
    engagement_shape_ratio: number;
  };
  speech: { detected_share: number; top_langs: LangShare[] };
  caption: { top_langs: LangShare[] };
};

type Props = { index: string; loaded?: Loaded };

export const SectionAudience = ({ index, loaded }: Props) => (
  <section>
    <SectionHeading index={index}>Audience</SectionHeading>
    {loaded ? (
      <>
        <dl style={kvGrid}>
          <Row label="followers" value={loaded.follower_bucket} />
          <Row label="median plays" value={formatCompact(loaded.posting.median_plays)} />
          <Row label="clip length" value={`${loaded.posting.median_clip_duration_s.toFixed(1)}s`} />
          <Row label="clips / week" value={loaded.posting.median_clips_per_week.toFixed(1)} />
          <Row label="engagement shape" value={`${loaded.posting.engagement_shape_ratio.toFixed(1)}×`} />
        </dl>
        <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          <div>
            <div style={subhead}>speech {Math.round(loaded.speech.detected_share * 100)}%</div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {loaded.speech.top_langs.map((l) => (
                <LangChip key={l.code} code={l.code} share={l.share} />
              ))}
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={subhead}>captions</div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
              {loaded.caption.top_langs.map((l) => (
                <LangChip key={l.code} code={l.code} share={l.share} />
              ))}
            </div>
          </div>
        </div>
      </>
    ) : (
      <>
        {[0,1,2,3,4].map((i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 12, marginBottom: 6 }}>
            <Skeleton height={10} />
            <Skeleton height={10} width={40} />
          </div>
        ))}
      </>
    )}
  </section>
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

const subhead: CSSProperties = {
  fontSize: 10, color: tokens.ink.faint, textTransform: "uppercase",
  letterSpacing: "0.10em", marginBottom: 5,
};
