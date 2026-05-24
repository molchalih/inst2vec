import { useAtomValue } from "jotai";
import { activeRunAtom, hoverAtom, type HoverState } from "@/state";
import type { AtlasRun } from "@/data";
import { Tooltip } from "@/ui";

const resolveLabel = (hover: HoverState, run: AtlasRun | null): string | null => {
  if (!run) return null;
  if (hover.clusterId !== null) {
    const c = run.clusters.find((cc) => cc.id === hover.clusterId);
    return c ? c.label : null;
  }
  if (hover.dotId !== null) {
    const u = run.users.find(([id]) => id === hover.dotId);
    if (!u) return null;
    const clusterId = u[3];
    if (clusterId < 0) return null;
    const c = run.clusters.find((cc) => cc.id === clusterId);
    return c ? c.label : null;
  }
  return null;
};

export const HoverTooltip = () => {
  const hover = useAtomValue(hoverAtom);
  const run = useAtomValue(activeRunAtom);
  const label = resolveLabel(hover, run);
  const visible = label !== null;
  return (
    <Tooltip x={hover.screenX} y={hover.screenY} visible={visible}>
      {label}
    </Tooltip>
  );
};
