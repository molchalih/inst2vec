import { useEffect } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  activeRunAtom,
  clusterDetailFor, ensureClusterBundleAtom,
  clusterLabelFor, ensureClusterLabelAtom,
} from "@/state";
import { clusterSummaryLede } from "@/core";
import type { ClusterLabel } from "@/data";
import { tokens } from "@/ui/tokens";
import { SectionAudience } from "../ui/SectionAudience";
import { SectionMusical } from "../ui/SectionMusical";
import { SectionSpoken } from "../ui/SectionSpoken";
import { SectionTextual } from "../ui/SectionTextual";
import { SectionVisual } from "../ui/SectionVisual";
import { PaneHeader } from "../ui/PaneHeader";
import { PaneBody } from "../ui/PaneBody";
import { PaneUnavailable } from "../ui/PaneUnavailable";

// `ClusterLabel.modality` is a union-typed field, not a discriminated
// union, so TS won't narrow the whole label to a modality literal from a
// bare `=== "audio"` check. This guard performs the runtime check and
// returns the narrowed type, so the per-modality Section props (which
// accept only their own modality) catch any mis-routing at compile time.
const labelForModality = <M extends ClusterLabel["modality"]>(
  label: ClusterLabel | undefined,
  modality: M,
): (ClusterLabel & { modality: M }) | undefined =>
  label?.modality === modality
    ? (label as ClusterLabel & { modality: M })
    : undefined;

// Modalities whose tags render in the Visual section (everything but the two
// language-distribution sections, Spoken=audio and Textual=textual).
const isVisualModality = (
  m: ClusterLabel["modality"] | null,
): m is "visual" | "music" | "multimodal" =>
  m === "visual" || m === "music" || m === "multimodal";

type Props = { clusterId: number };

export const ClusterPane = ({ clusterId }: Props) => {
  const run = useAtomValue(activeRunAtom);
  const slot = useAtomValue(clusterDetailFor(clusterId));
  const labelSlot = useAtomValue(clusterLabelFor(clusterId));
  const ensureBundle = useSetAtom(ensureClusterBundleAtom);
  const ensureLabel = useSetAtom(ensureClusterLabelAtom);

  // Static fixtures only carry detail for clusters whose has_detail is true.
  const cluster = run?.clusters.find((c) => c.id === clusterId);
  const detailAvailable = !!run?.meta.details_available && !!cluster?.has_detail;

  const d = slot.data;
  // The heavy label (tags + summary) is deferred: fetch it on selection, but
  // only once the main detail tells us the cluster actually has one. Re-fire on
  // the committed-run id so a case switch that lands a new run reloads the label
  // (ensureClusterLabelAtom no-ops until the committed run catches up).
  const committedRunId = run?.meta.id ?? null;
  const labelModality = d?.label_modality ?? null;
  useEffect(() => {
    if (labelModality === null) return;
    ensureLabel(clusterId).catch(() => {});
  }, [clusterId, labelModality, committedRunId, ensureLabel]);

  if (!run) return null;
  if (!cluster) return null;

  const total = run.clusters.reduce((s, c) => s + (c.id >= 0 ? c.size : 0), 0);
  const pct = total > 0 ? (cluster.size / total) * 100 : 0;

  const meta = d
    ? `${cluster.size.toLocaleString()} creators · ${pct.toFixed(1)}% of case · ${d.activity_span_months} months active`
    : `${cluster.size.toLocaleString()} creators · ${pct.toFixed(1)}% of case`;
  // `labelSlot.label` is present (possibly `null`, meaning "loaded, no tags")
  // once resolved; absent while pending. Keep the null-vs-pending distinction so
  // a tagless cluster doesn't sit on the skeleton forever.
  const labelLoaded = labelSlot.label !== undefined;
  const label = labelSlot.label ?? undefined;
  const lede = label?.summary ? clusterSummaryLede(label.summary) : undefined;
  // The label is still in flight: the cluster has one, but it's neither loaded
  // (incl. loaded-null) nor errored yet.
  const labelLoading = labelModality !== null && !labelLoaded && !labelSlot.error;
  const visualModality = isVisualModality(labelModality) ? labelModality : undefined;

  if (!detailAvailable) {
    return (
      <PaneBody>
        <PaneHeader name={cluster.label} meta={meta} lede={lede} />
        <PaneUnavailable />
      </PaneBody>
    );
  }

  if (slot.error) {
    return (
      <PaneBody>
        <PaneHeader name={cluster.label} meta={meta} lede={lede} />
        <FetchError onRetry={() => { ensureBundle().catch(() => {}); }} />
      </PaneBody>
    );
  }

  return (
    <PaneBody fill>
      <PaneHeader name={cluster.label} meta={meta} lede={lede} />
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
          <SectionSpoken
            index="03"
            loaded={d.speech}
            label={labelForModality(label, "audio")}
            labelLoading={labelModality === "audio" && labelLoading}
          />
          <SectionTextual
            index="04"
            loaded={d.caption}
            label={labelForModality(label, "textual")}
            labelLoading={labelModality === "textual" && labelLoading}
          />
          {/* Audio/textual tags render in their own sections above; the Visual
              section carries visual / music / multimodal tags (and their
              loading skeleton). */}
          <SectionVisual
            index="05"
            cluster={visualModality ? label : undefined}
            clusterLoading={visualModality !== undefined && labelLoading}
            modality={visualModality}
          />
        </>
      ) : (
        <>
          <SectionAudience index="01" />
          <SectionMusical  index="02" />
          <SectionSpoken   index="03" />
          <SectionTextual  index="04" />
          <SectionVisual   index="05" />
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
