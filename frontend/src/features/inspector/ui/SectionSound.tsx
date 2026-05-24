import type { CSSProperties } from "react";
import { phraseFor } from "@/core";
import type { AudioScores, WeightedTag, DistinctivenessEntry } from "@/data";
import { AudioBar } from "./primitives/AudioBar";
import { Chip } from "./primitives/Chip";
import { Skeleton } from "./primitives/Skeleton";
import { SectionHeading } from "./SectionHeading";

type Loaded = {
  audio: AudioScores;
  genreTop: WeightedTag[];
  instrumentTop: WeightedTag[];
  distinctiveness: DistinctivenessEntry[];
};

type SectionSoundProps = {
  /** Catalogue index, e.g. "01". */
  index: string;
  /** Loaded detail slice, or undefined while loading. */
  loaded?: Loaded;
};

export const SectionSound = ({ index, loaded }: SectionSoundProps) => (
  <section>
    <SectionHeading index={index}>Sound</SectionHeading>
    {loaded ? (
      <>
        <AudioBar name="approachability" value={loaded.audio.approachability} />
        <AudioBar name="engagement"      value={loaded.audio.engagement} />
        <AudioBar name="danceability"    value={loaded.audio.danceability} />
        <div style={chipRow}>
          {[...loaded.genreTop, ...loaded.instrumentTop].map((t) => (
            <Chip key={t.label} weight={t.weight}>{t.label}</Chip>
          ))}
          {loaded.distinctiveness.map((e) => {
            const p = phraseFor(e);
            return <Chip key={e.feature}>{p.arrow} {p.label}</Chip>;
          })}
        </div>
      </>
    ) : (
      <>
        <Skeleton height={6} /><div style={{ height: 6 }} />
        <Skeleton height={6} /><div style={{ height: 6 }} />
        <Skeleton height={6} /><div style={{ height: 10 }} />
        <Skeleton height={22} />
      </>
    )}
  </section>
);

const chipRow: CSSProperties = {
  display: "flex", flexWrap: "wrap", gap: 5, marginTop: 10,
};
