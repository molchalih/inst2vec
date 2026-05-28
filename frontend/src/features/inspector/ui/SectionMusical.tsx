import type { CSSProperties } from "react";
import { phraseFor } from "@/core";
import { tokens } from "@/ui/tokens";
import type {
  AudioScores,
  DistinctivenessEntry,
  MoodShares,
  TimbreShares,
  WeightedTag,
} from "@/data";
import { AudioBar } from "./primitives/AudioBar";
import { Chip } from "./primitives/Chip";
import { MicroBar } from "./primitives/MicroBar";
import { Skeleton } from "./primitives/Skeleton";
import { CollapsibleSection } from "./CollapsibleSection";

type Loaded = {
  audio: AudioScores;
  mood: MoodShares;
  timbre: TimbreShares;
  genreTop: WeightedTag[];
  instrumentTop: WeightedTag[];
  distinctiveness: DistinctivenessEntry[];
};

type Props = {
  /** Catalogue index, e.g. "02". */
  index: string;
  /** Loaded detail slice, or undefined while loading. */
  loaded?: Loaded;
};

const moodOrder: (keyof MoodShares)[] = ["party", "happy", "relaxed", "aggressive", "sad"];
const moodLabel: Record<keyof MoodShares, string> = {
  party: "party", happy: "happy", relaxed: "relaxed",
  aggressive: "aggressive", sad: "sad",
};

const timbreOrder: (keyof TimbreShares)[] = [
  "tonal", "electronic", "bright", "female_voice", "acoustic",
];
const timbreLabel: Record<keyof TimbreShares, string> = {
  tonal: "tonal", electronic: "electronic", bright: "bright",
  female_voice: "fem. voice", acoustic: "acoustic",
  instrumental: "instrumental",
};

/**
 * Combined musical section: three sub-groups separated by tiny
 * tracked-out subheadings — engagement (audio bars), character (mood
 * + timbre MicroBars), and tags (genre / instrument / distinctiveness
 * chips).
 */
export const SectionMusical = ({ index, loaded }: Props) => (
  <CollapsibleSection index={index} title="Musical">
    {loaded ? (
      <>
        <Subhead>Engagement</Subhead>
        <AudioBar name="approachability" value={loaded.audio.approachability} />
        <AudioBar name="engagement"      value={loaded.audio.engagement} />
        <AudioBar name="danceability"    value={loaded.audio.danceability} />

        <Subhead>Character</Subhead>
        <div style={shareGrid}>
          <div>
            {moodOrder.map((k) => (
              <MicroBar
                key={k}
                name={moodLabel[k]}
                value={loaded.mood[k]}
                color={tokens.inspector.microBar.moodColor}
              />
            ))}
          </div>
          <div>
            {timbreOrder.map((k) => (
              <MicroBar
                key={k}
                name={timbreLabel[k]}
                value={loaded.timbre[k]}
                color={tokens.inspector.microBar.timbreColor}
              />
            ))}
          </div>
        </div>

        <Subhead>Tags</Subhead>
        <div style={chipRow}>
          {loaded.distinctiveness.map((e) => {
            const p = phraseFor(e);
            return <Chip key={e.feature}>{p.arrow} {p.label}</Chip>;
          })}
          {[...loaded.genreTop, ...loaded.instrumentTop].map((t) => (
            <Chip key={t.label} weight={t.weight}>{t.label}</Chip>
          ))}
        </div>
      </>
    ) : (
      <>
        <Skeleton height={6} /><div style={{ height: 6 }} />
        <Skeleton height={6} /><div style={{ height: 6 }} />
        <Skeleton height={6} /><div style={{ height: 10 }} />
        <div style={shareGrid}>
          <div>{[0,1,2,3,4].map((i) => <div key={i} style={{ marginBottom: 6 }}><Skeleton height={5} /></div>)}</div>
          <div>{[0,1,2,3,4].map((i) => <div key={i} style={{ marginBottom: 6 }}><Skeleton height={5} /></div>)}</div>
        </div>
        <Skeleton height={22} />
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

const shareGrid: CSSProperties = {
  display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14,
};

const chipRow: CSSProperties = {
  display: "flex", flexWrap: "wrap", gap: 5,
};
