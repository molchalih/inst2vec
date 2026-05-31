import { useCallback, useLayoutEffect, useRef } from "react";
import type { Graphics as PixiGraphics } from "pixi.js";
import { useAtomValue } from "jotai";
import {
  trackedCreatorAtom, stretchedRunAtom, viewportAtom,
  useTransition, useMorphJoin,
} from "@/state";
import { flightProgress, interpolatedUserPos, easeOutCubic } from "@/core";
import { useSinePulse, useEasedScalar } from "@/interaction";
import { tokens } from "@/ui/tokens";
import { drawTrackingHalo } from "../draw/drawTrackingHalo";
import { drawTrackingMarker } from "../draw/drawTrackingMarker";

const { flightMs, pulseMs } = tokens.motion.versionSwitch.phase3;
const flightFrac = flightMs / (flightMs + pulseMs);

/**
 * Persistent tracked-dot beacon overlay. A soft white halo plus a breathing
 * white border ring around the NORMAL-size dot (the base dot is the cluster
 * colour), both riding one continuous sine. The whole beacon fades in on track
 * and out on untrack via an eased visibility scalar: a `displayedCreatorId` ref
 * holds the last tracked id so the layer keeps drawing the LAST position while
 * the eased value decays to 0 (mirrors HoverLayer's displayedDotId). At rest it
 * resolves the dot from the active run by creator id; DURING a version-switch
 * morph it stays glued to the creator's interpolated position via the shared
 * morph join + the same flight-progress math DotsLayer uses. Mounts two
 * graphics: the soft halo at zIndex 0.5 (beneath the dots layer at 1, so it
 * reads as a glow), and the breathing border at zIndex 1.5 (above dots, so the
 * ring isn't occluded by neighbours).
 */
export const TrackingLayer = () => {
  const trackedCreatorId = useAtomValue(trackedCreatorAtom);
  const run = useAtomValue(stretchedRunAtom);
  const viewport = useAtomValue(viewportAtom);
  const transition = useTransition();
  const joined = useMorphJoin();

  const displayedCreatorId = useRef<number | null>(null);
  useLayoutEffect(() => {
    if (trackedCreatorId !== null) displayedCreatorId.current = trackedCreatorId;
  }, [trackedCreatorId]);

  const vis = useEasedScalar(trackedCreatorId !== null, tokens.motion.slow, easeOutCubic);
  // Keep the breathing rAF alive while anything is still visible — including the
  // fade-out window where trackedCreatorId is null but `vis` has not yet decayed
  // to 0 — so the sine never freezes mid-fade.
  const pulse = useSinePulse(vis > 0, tokens.track.pulse.periodMs, 0, 1);

  const resolvePos = useCallback((): readonly [number, number] | null => {
    const id = trackedCreatorId ?? displayedCreatorId.current;
    if (id === null) return null;
    if (transition && joined) {
      const motion = flightProgress(transition.phase, transition.progress, flightFrac);
      return interpolatedUserPos(joined, id, motion);
    }
    const u = run?.users.find(([uid]) => uid === id);
    return u ? [u[1], u[2]] : null;
  }, [trackedCreatorId, transition, joined, run]);

  const drawHalo = useCallback(
    (g: PixiGraphics) => {
      drawTrackingHalo(g, resolvePos(), viewport, pulse, vis);
    },
    [resolvePos, viewport, pulse, vis],
  );

  const drawMarker = useCallback(
    (g: PixiGraphics) => {
      drawTrackingMarker(g, resolvePos(), viewport, pulse, vis);
    },
    [resolvePos, viewport, pulse, vis],
  );

  return (
    <>
      <pixiGraphics draw={drawHalo} zIndex={0.5} />
      <pixiGraphics draw={drawMarker} zIndex={1.5} />
    </>
  );
};
