import { useEffect } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  activeRunAtom, clusterDetailFor, ensureClusterDetailAtom,
} from "@/state";
import { tokens } from "@/ui/tokens";
import { SectionSound } from "../ui/SectionSound";
import { SectionCharacter } from "../ui/SectionCharacter";
import { SectionAudience } from "../ui/SectionAudience";
import { SectionWhereItSits } from "../ui/SectionWhereItSits";
import { PaneHeader } from "../ui/PaneHeader";
import { PaneBody } from "../ui/PaneBody";
import { PaneUnavailable } from "../ui/PaneUnavailable";

type Props = { clusterId: number };

export const ClusterPane = ({ clusterId }: Props) => {
  const run = useAtomValue(activeRunAtom);
  const slot = useAtomValue(clusterDetailFor(clusterId));
  const ensure = useSetAtom(ensureClusterDetailAtom);

  // Static fixtures only ship runs/{runId}/clusters/{id}.json when the
  // run's details_available is true and the cluster's has_detail is
  // true. Fetching otherwise is a guaranteed 404, which would replace
  // the pane with the FetchError UI; skip the load and render the
  // basic-metadata fallback instead.
  const cluster = run?.clusters.find((c) => c.id === clusterId);
  const detailAvailable = !!run?.meta.details_available && !!cluster?.has_detail;
  useEffect(() => {
    if (!detailAvailable) return;
    void ensure(clusterId).catch(() => {});
  }, [clusterId, ensure, detailAvailable]);

  if (!run) return null;
  if (!cluster) return null;

  const total = run.clusters.reduce((s, c) => s + (c.id >= 0 ? c.size : 0), 0);
  const pct = total > 0 ? (cluster.size / total) * 100 : 0;

  const meta = slot.data
    ? `${cluster.size.toLocaleString()} creators · ${pct.toFixed(1)}% of run · ${slot.data.activity_span_months} months active`
    : `${cluster.size.toLocaleString()} creators · ${pct.toFixed(1)}% of run`;

  if (!detailAvailable) {
    return (
      <PaneBody>
        <PaneHeader name={cluster.label} meta={meta} />
        <PaneUnavailable />
      </PaneBody>
    );
  }

  if (slot.error) {
    return (
      <PaneBody>
        <PaneHeader name={cluster.label} meta={meta} />
        <FetchError onRetry={() => void ensure(clusterId).catch(() => {})} />
      </PaneBody>
    );
  }

  const d = slot.data;
  return (
    <PaneBody fill>
      <PaneHeader name={cluster.label} meta={meta} />
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
          <SectionWhereItSits index="04" kind="cluster" loaded={{ nearest_clusters: d.spatial.nearest_clusters }} />
        </>
      ) : (
        <>
          <SectionSound index="01" />
          <SectionCharacter index="02" />
          <SectionAudience index="03" />
          <SectionWhereItSits index="04" kind="cluster" />
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
