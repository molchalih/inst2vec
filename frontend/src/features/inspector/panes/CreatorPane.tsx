import { useEffect } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  activeRunAtom, creatorDetailFor, ensureCreatorDetailAtom,
} from "@/state";
import type { ClusterLabel, EmbeddingCase } from "@/data";
import { tokens } from "@/ui/tokens";
import { SectionAudience } from "../ui/SectionAudience";
import { SectionMusical } from "../ui/SectionMusical";
import { SectionSpoken } from "../ui/SectionSpoken";
import { SectionTextual } from "../ui/SectionTextual";
import { SectionVisual } from "../ui/SectionVisual";
import { PaneHeader } from "../ui/PaneHeader";
import { PaneBody } from "../ui/PaneBody";
import { PaneUnavailable } from "../ui/PaneUnavailable";

// Map the embedding case keyed on the run to the modality string that drives
// SectionVisual's section heading. Without this, the creator pane (which has
// no cluster context) would render every per-clip section titled "Visual"
// regardless of the active case.
const MODALITY_FOR_CASE: Record<EmbeddingCase, ClusterLabel["modality"]> = {
  video: "visual",
  sandwich: "multimodal",
  audio: "audio",
  maest: "music",
};

type Props = { creatorId: number };

export const CreatorPane = ({ creatorId }: Props) => {
  const run = useAtomValue(activeRunAtom);
  const slot = useAtomValue(creatorDetailFor(creatorId));
  const ensure = useSetAtom(ensureCreatorDetailAtom);

  // Detail fixtures only exist when the run advertises details and the
  // user's own has_detail flag is true. Deep-link URLs (#user=N) can
  // bypass selectDotAtom's guard, so this check is the load-side gate.
  const user = run?.users.find(([id]) => id === creatorId);
  const detailAvailable = !!run?.meta.details_available && user?.[4] === true;
  useEffect(() => {
    if (!detailAvailable) return;
    void ensure(creatorId).catch(() => {});
  }, [creatorId, ensure, detailAvailable]);

  if (!run) return null;
  if (!user) return null;
  const name = `user #${creatorId}`;

  if (!detailAvailable) {
    return (
      <PaneBody>
        <PaneHeader name={name} meta="" />
        <PaneUnavailable />
      </PaneBody>
    );
  }

  const meta = slot.data
    ? `${slot.data.n_clips} clips · ${slot.data.follower_bucket} · ${slot.data.activity_span_months} months active`
    : "loading…";

  if (slot.error) {
    return (
      <PaneBody>
        <PaneHeader name={name} meta={meta} />
        <FetchError onRetry={() => void ensure(creatorId).catch(() => {})} />
      </PaneBody>
    );
  }

  const d = slot.data;
  const modality = MODALITY_FOR_CASE[run.meta.case];
  return (
    <PaneBody fill>
      <PaneHeader name={name} meta={meta} />
      {d ? (
        <>
          <SectionAudience
            index="01"
            loaded={{
              follower_bucket: d.follower_bucket,
              posting: d.posting,
            }}
          />
          <SectionMusical
            index="02"
            loaded={{
              audio: d.audio,
              mood: d.mood_shares,
              timbre: d.timbre_shares,
              genreTop: d.genre_top,
              instrumentTop: d.instrument_top,
              distinctiveness: d.distinctiveness,
            }}
          />
          <SectionSpoken  index="03" loaded={d.speech} />
          <SectionTextual index="04" loaded={d.caption} />
          <SectionVisual  index="05" clips={d.clips} modality={modality} />
        </>
      ) : (
        <>
          <SectionAudience index="01" />
          <SectionMusical  index="02" />
          <SectionSpoken   index="03" />
          <SectionTextual  index="04" />
          <SectionVisual   index="05" clips={[]} modality={modality} />
        </>
      )}
    </PaneBody>
  );
};

const FetchError = ({ onRetry }: { onRetry: () => void }) => (
  <div style={{
    marginTop: tokens.inspector.sectionHead.gapTop,
    fontSize: 12, color: tokens.ink.muted, fontFamily: tokens.type.mono,
    padding: 10, background: "rgb(255 255 255 / 0.03)", borderRadius: 6,
    display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10,
  }}>
    <span>Couldn't load detail</span>
    <button type="button" onClick={onRetry} style={{
      background: "none", border: `1px solid ${tokens.ink.line}`,
      color: tokens.ink.default, borderRadius: 4, padding: "3px 8px",
      fontSize: 11, fontFamily: tokens.type.mono, cursor: "pointer",
    }}>retry</button>
  </div>
);
