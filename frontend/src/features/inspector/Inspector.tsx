import { useEffect, useRef, useState, type CSSProperties } from "react";
import { useAtomValue } from "jotai";
import { activeRunAtom, selectionAtom, useClearSelection, type Selection } from "@/state";
import { colorForCluster } from "@/core";
import { Panel } from "@/ui";
import { tokens } from "@/ui/tokens";
import { ClusterPane } from "./panes/ClusterPane";
import { CreatorPane } from "./panes/CreatorPane";
import { PaneShell } from "./ui/PaneShell";

type ActiveSelection = NonNullable<Selection>;

export const Inspector = () => {
  const sel = useAtomValue(selectionAtom);
  const run = useAtomValue(activeRunAtom);
  const close = useClearSelection();
  const lastSelRef = useRef<ActiveSelection | null>(null);
  const [displayedKey, setDisplayedKey] = useState<string | null>(null);
  const [fadeIn, setFadeIn] = useState(true);

  if (sel) lastSelRef.current = sel;
  const displayed = sel ?? lastSelRef.current;
  const nextKey = displayed
    ? `${displayed.kind}-${displayed.kind === "cluster" ? displayed.clusterId : displayed.creatorId}`
    : null;

  // Content crossfade when the displayed key changes while open.
  // First open (displayedKey === null) has no current content to fade
  // out, so seed directly with fadeIn=true to avoid leaving the pane
  // transparent.
  useEffect(() => {
    if (!sel) return;
    if (nextKey === displayedKey) return;
    if (displayedKey === null) {
      setDisplayedKey(nextKey);
      setFadeIn(true);
      return;
    }
    setFadeIn(false);
    const t = window.setTimeout(() => {
      setDisplayedKey(nextKey);
      setFadeIn(true);
    }, tokens.inspector.crossfadeMs);
    return () => window.clearTimeout(t);
  }, [nextKey, sel, displayedKey]);

  if (!displayed) return null;

  // Resolve the pane's accent here (not in each pane) so the whole
  // panel — gradient and bars — shares one `--accent`. A creator
  // borrows its parent cluster's colour.
  const clusterId = displayed.kind === "cluster"
    ? displayed.clusterId
    : run?.users.find(([id]) => id === displayed.creatorId)?.[3];
  const accent = colorForCluster(clusterId ?? -1, tokens.palette.cluster, tokens.palette.noise);

  return (
    <Panel open={sel != null} side="left" onClose={close}>
      <div style={{ "--accent": accent, ...stage } as CSSProperties}>
        <div aria-hidden="true" style={washStyle} />
        <PaneShell onClose={close}>
          <div style={{
            ...crossfade,
            opacity: fadeIn ? 1 : 0,
            transition: `opacity ${tokens.inspector.crossfadeMs}ms ease`,
          }}>
            {displayed.kind === "cluster"
              ? <ClusterPane clusterId={displayed.clusterId} />
              : <CreatorPane creatorId={displayed.creatorId} />}
          </div>
        </PaneShell>
      </div>
    </Panel>
  );
};

// Fills the panel and is the containing block for the wash layer.
const stage: CSSProperties = {
  position: "relative", flex: 1, minHeight: 0,
  display: "flex", flexDirection: "column",
};

// Full-bleed top→bottom accent wash behind the content.
const washStyle: CSSProperties = {
  position: "absolute", inset: 0, pointerEvents: "none",
  background: `linear-gradient(to bottom, color-mix(in srgb, var(--accent) ${tokens.inspector.wash.topPct}%, transparent), transparent ${tokens.inspector.wash.fadeStop}%)`,
};

const crossfade: CSSProperties = {
  flex: 1, minHeight: 0, display: "flex", flexDirection: "column",
};
