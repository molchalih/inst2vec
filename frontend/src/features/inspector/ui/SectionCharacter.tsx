import { tokens } from "@/ui/tokens";
import type { MoodShares, TimbreShares } from "@/data";
import { MicroBar } from "./primitives/MicroBar";
import { Skeleton } from "./primitives/Skeleton";
import { SectionHeading } from "./SectionHeading";

type Loaded = { mood: MoodShares; timbre: TimbreShares };
type SectionCharacterProps = { index: string; loaded?: Loaded };

const moodOrder: (keyof MoodShares)[] = ["party", "happy", "relaxed", "aggressive", "sad"];
const moodLabel: Record<keyof MoodShares, string> = {
  party: "party", happy: "happy", relaxed: "relaxed",
  aggressive: "aggressive", sad: "sad",
};
const timbreOrder: (keyof TimbreShares)[] = ["tonal", "electronic", "bright", "female_voice", "acoustic"];
const timbreLabel: Record<keyof TimbreShares, string> = {
  tonal: "tonal", electronic: "electronic", bright: "bright",
  female_voice: "fem. voice", acoustic: "acoustic",
  instrumental: "instrumental",
};

export const SectionCharacter = ({ index, loaded }: SectionCharacterProps) => (
  <section>
    <SectionHeading index={index}>Character</SectionHeading>
    {loaded ? (
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <div>
          {moodOrder.map((k) => (
            <MicroBar key={k} name={moodLabel[k]} value={loaded.mood[k]}
                      color={tokens.inspector.microBar.moodColor} />
          ))}
        </div>
        <div>
          {timbreOrder.map((k) => (
            <MicroBar key={k} name={timbreLabel[k]} value={loaded.timbre[k]}
                      color={tokens.inspector.microBar.timbreColor} />
          ))}
        </div>
      </div>
    ) : (
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <div>{[0,1,2,3,4].map((i) => <div key={i} style={{ marginBottom: 6 }}><Skeleton height={5} /></div>)}</div>
        <div>{[0,1,2,3,4].map((i) => <div key={i} style={{ marginBottom: 6 }}><Skeleton height={5} /></div>)}</div>
      </div>
    )}
  </section>
);
