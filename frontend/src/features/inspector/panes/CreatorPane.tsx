import { useEffect } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  activeRunAtom, creatorDetailFor, ensureCreatorDetailAtom,
} from "@/state";
import { tokens } from "@/ui/tokens";
import { SectionSound } from "../ui/SectionSound";
import { SectionCharacter } from "../ui/SectionCharacter";
import { SectionAudience } from "../ui/SectionAudience";
import { SectionWhereItSits } from "../ui/SectionWhereItSits";
import { PaneHeader } from "../ui/PaneHeader";
import { PaneBody } from "../ui/PaneBody";
import { PaneUnavailable } from "../ui/PaneUnavailable";

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
  return (
    <PaneBody fill>
      <PaneHeader name={name} meta={meta} />
      {d ? (
        <>
          <SectionSound
            index="01"
            loaded={{
              audio: d.audio,
              genreTop: d.genre_top,
              instrumentTop: d.instrument_top,
              distinctiveness: d.distinctiveness,
            }}
          />
          <SectionCharacter index="02" loaded={{ mood: d.mood_shares, timbre: d.timbre_shares }} />
          <SectionAudience index="03" loaded={{
            follower_bucket: d.follower_bucket,
            posting: d.posting,
            speech: d.speech,
            caption: d.caption,
          }} />
          <SectionWhereItSits index="04" kind="creator" loaded={{ nearest_other_cluster: d.spatial.nearest_other_cluster }} />
        </>
      ) : (
        <>
          <SectionSound index="01" />
          <SectionCharacter index="02" />
          <SectionAudience index="03" />
          <SectionWhereItSits index="04" kind="creator" />
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
