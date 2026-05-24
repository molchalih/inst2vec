import { useEffect } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  activeRunIdAtom, ensureManifestAtom, ensureRunAtom, useIsTransitioning,
} from "@/state";
import {
  useFitOnActiveRun, useUrlSync, useTrackViewportSize, useVersionTransition,
  useIntroAnimation,
} from "@/interaction";
import { Stage, DotsLayer, EllipsesLayer, HoverLayer } from "@/render";
import { Providers } from "./providers";
import {
  VersionsFeature, SelectionFeature, InspectorFeature, SearchFeature,
  HoverTooltipFeature, useCameraFocus,
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
    void ensureManifest().catch((err: unknown) => {
      console.error("ensureManifest failed", err);
    });
  }, [ensureManifest]);

  useEffect(() => {
    if (!runId || isTransitioning) return;
    void ensureRun(runId).catch((err: unknown) => {
      console.error("ensureRun failed", err);
    });
  }, [runId, isTransitioning, ensureRun]);

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

export const AppShell = () => (
  <Providers>
    <Routing />
    <ViewportTracker />
    <RunLoader />
    <Fitting />
    <Intro />
    <VersionTransition />
    <CameraFocus />
    <main className="w-screen h-screen bg-bg-canvas relative">
      <Stage>
        <EllipsesLayer />
        <DotsLayer />
        <HoverLayer />
      </Stage>
      <VersionsFeature />
      <HoverTooltipFeature />
      <SelectionFeature />
      <InspectorFeature />
      <SearchFeature />
    </main>
  </Providers>
);
