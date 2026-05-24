import { useSelectCluster } from "@/state";
import { colorForCluster } from "@/core";
import { tokens } from "@/ui/tokens";
import type { NearestCluster, NearestOtherCluster } from "@/data";
import { NeighborRow } from "./primitives/NeighborRow";
import { Skeleton } from "./primitives/Skeleton";
import { SectionHeading } from "./SectionHeading";

type Props = { index: string } & (
  | { kind: "cluster"; loaded?: { nearest_clusters: NearestCluster[] } }
  | { kind: "creator"; loaded?: { nearest_other_cluster: NearestOtherCluster | null } }
);

const distanceLabel = (i: number, total: number): string => {
  if (i === 0) return "closest";
  if (i === total - 1) return "farther";
  return "nearby";
};

export const SectionWhereItSits = (props: Props) => {
  const selectCluster = useSelectCluster();

  if (!props.loaded) {
    return (
      <section>
        <SectionHeading index={props.index}>Surroundings</SectionHeading>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {[0, 1, 2].map((i) => <Skeleton key={i} height={26} />)}
        </div>
      </section>
    );
  }

  const rows = props.kind === "cluster"
    ? props.loaded.nearest_clusters
    : props.loaded.nearest_other_cluster ? [props.loaded.nearest_other_cluster] : [];

  if (rows.length === 0) return null;

  const colorOf = (clusterId: number): string =>
    colorForCluster(clusterId, tokens.palette.cluster, tokens.palette.noise);

  return (
    <section>
      <SectionHeading index={props.index}>Surroundings</SectionHeading>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {rows.map((r, i) => (
          <NeighborRow
            key={r.cluster_id}
            label={r.label}
            color={colorOf(r.cluster_id)}
            distanceLabel={
              props.kind === "creator" ? "near (borderline)" : distanceLabel(i, rows.length)
            }
            onClick={() => selectCluster(r.cluster_id)}
          />
        ))}
      </div>
    </section>
  );
};
