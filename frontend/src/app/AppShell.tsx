import { useEffect } from "react";
import { useAtomValue, useSetAtom, useStore } from "jotai";
import {
  activeRunIdAtom, activeRunAtom, ensureManifestAtom, ensureRunAtom, prefetchRunAtom,
  ensureClusterBundleAtom,
  manifestAtom, runStateAtom, trackedCreatorAtom, useIsTransitioning,
  useIsChromeRevealed,
} from "@/state";
import {
  useFitOnActiveRun, useUrlSync, useTrackViewportSize, useVersionTransition,
  useIntroAnimation, useCaseSwitchOrchestrator,
} from "@/interaction";
import { Stage, DotsLayer, EllipsesLayer, HoverLayer, TrackingLayer } from "@/render";
import { ControlDock } from "@/ui";
import { Providers } from "./providers";
import {
  VersionsFeature, SelectionFeature, InspectorFeature, SearchFeature,
  HoverTooltipFeature, TrackingFeature, DocsLinkFeature, useCameraFocus,
} from "@/features";

const RunLoader = () => {
  const ensureManifest = useSetAtom(ensureManifestAtom);
  const ensureRun = useSetAtom(ensureRunAtom);
  const runId = useAtomValue(activeRunIdAtom);
  // Programmatic route updates (hashchange, deep-link) can change runId
  // mid-transition; ensureRunAtom's writer guard would drop those requests
  // and they'd never retry because runId wouldn't change again. Re-fire on
  // transition-clear so the dropped request lands once the guard releases.
  const isTransitioning = useIsTransitioning();

  useEffect(() => {
    ensureManifest().catch((err: unknown) => {
      console.error("ensureManifest failed", err);
    });
  }, [ensureManifest]);

  useEffect(() => {
    if (!runId || isTransitioning) return;
    ensureRun(runId).catch((err: unknown) => {
      console.error("ensureRun failed", err);
    });
  }, [runId, isTransitioning, ensureRun]);

  return null;
};

// Eager presence prefetch. When a creator becomes tracked, load every other
// manifest run into the cache via prefetchRunAtom (cache-only — never drives a
// version switch) so the presence gate converges to exact. Idempotent: cached
// runs are skipped; the active run is already loaded.
const TrackingPrefetch = () => {
  const trackedCreatorId = useAtomValue(trackedCreatorAtom);
  const manifest = useAtomValue(manifestAtom);
  const prefetchRun = useSetAtom(prefetchRunAtom);
  const store = useStore();

  useEffect(() => {
    if (trackedCreatorId === null || !manifest) return;
    const cached = store.get(runStateAtom).runs;
    for (const run of manifest.runs) {
      if (cached.has(run.id)) continue;
      prefetchRun(run.id).catch((err: unknown) => {
        console.error("prefetchRun failed", err);
      });
    }
  }, [trackedCreatorId, manifest, prefetchRun, store]);

  return null;
};

// Eager cluster main-detail prefetch. Keys off the *committed* run
// (`activeRunAtom`), not the intended `activeRunIdAtom`: the API client resolves
// the committed run, so firing before the bulk run commits would fetch against a
// not-yet-active run and error. On first run load and on every pill switch the
// committed run id changes, warming the active run's `clusters-detail` bundle so
// clicking a cluster shows everything-but-tags instantly. Non-blocking: the dots
// paint regardless; failures are logged, never surfaced. Tags stay deferred —
// they load per-cluster on selection (see ClusterPane).
const ClusterDetailPrefetch = () => {
  const committedRunId = useAtomValue(activeRunAtom)?.meta.id ?? null;
  const ensureBundle = useSetAtom(ensureClusterBundleAtom);

  useEffect(() => {
    if (!committedRunId) return;
    ensureBundle().catch((err: unknown) => {
      console.error("ensureClusterBundle failed", err);
    });
  }, [committedRunId, ensureBundle]);

  return null;
};

const Routing = () => {
  useUrlSync();
  return null;
};

const ViewportTracker = () => {
  useTrackViewportSize();
  return null;
};

const Fitting = () => {
  useFitOnActiveRun();
  return null;
};

const VersionTransition = () => {
  useVersionTransition();
  return null;
};

const Intro = () => {
  useIntroAnimation();
  return null;
};

const CameraFocus = () => {
  useCameraFocus();
  return null;
};

const CaseSwitchOrchestrator = () => {
  useCaseSwitchOrchestrator();
  return null;
};

// Hosts the right-edge glyph controls and gates their one-time entrance on the
// chrome-reveal signal (intro flight settled). Order is dock order: tracking
// pins the bottom corner, the paper link stacks above it.
const Dock = () => {
  const revealed = useIsChromeRevealed();
  return (
    <ControlDock revealed={revealed}>
      <TrackingFeature />
      <DocsLinkFeature />
    </ControlDock>
  );
};

export const AppShell = () => (
  <Providers>
    <Routing />
    <ViewportTracker />
    <RunLoader />
    <TrackingPrefetch />
    <ClusterDetailPrefetch />
    <Fitting />
    <Intro />
    <VersionTransition />
    <CameraFocus />
    <CaseSwitchOrchestrator />
    <main className="w-screen h-screen bg-bg-canvas relative">
      <Stage>
        <EllipsesLayer />
        <DotsLayer />
        <TrackingLayer />
        <HoverLayer />
      </Stage>
      <VersionsFeature />
      <HoverTooltipFeature />
      <SelectionFeature />
      <InspectorFeature />
      <SearchFeature />
      <Dock />
    </main>
  </Providers>
);
