import { useCallback, useMemo } from "react";
import type { Graphics as PixiGraphics } from "pixi.js";
import {
  useStretchedRun, useViewport, useTransition, useIntro,
} from "@/state";
import {
  joinUsersByCreator, interpolateUsers,
  computeWaveDelays, waveProgress, scalePop,
  easeOutCubic, centralityRadiusScale,
  type JoinedUser,
} from "@/core";
import { tokens } from "@/ui/tokens";
import { drawDots } from "../draw/drawDots";
import {
  runToDotsFrame, runToIntroDotsFrame, type DotsFrame, type DrawableUser,
} from "../frame";

// Absolute per-side alpha: signal users render at tokens.dot.alpha, noise at
// tokens.noise.alpha. Missing side defaults to the present side's value so
// one-side creators don't snap when the present side is noise.
const sideAlpha = (cluster: number | null, fallback: number | null): number => {
  const id = cluster ?? fallback;
  if (id === null) return tokens.dot.alpha;
  return id < 0 ? tokens.noise.alpha : tokens.dot.alpha;
};

const morphFrame = (
  joined: ReadonlyArray<JoinedUser>,
  delayNorm: Float32Array,
  phase: 0 | 1 | 2 | 3,
  progress: number,
): DotsFrame => {
  // Phase 0/1: freeze every dot at from-side. Phase 3: freeze at to-side.
  // Phase 2: split into uniform flight (motion) then per-dot wave-pulse
  // (color + scalePop). Per-dot wave delays come from toXY distance, so
  // the pulse fans outward from the new arrangement's centre.
  // We short-circuit by PHASE rather than by a clamped globalMorph: with
  // non-zero jitter, the per-dot window can extend past [0, 1] (start < 0
  // or end > 1), so calling waveProgress at the boundary leaks a non-
  // zero/non-one value into phases 0/1/3 and the from/to sides never
  // quite land.
  const { spread, jitter, scalePopPeak, scalePopUpFrac } = tokens.motion.wave;
  // Flight and pulse durations are independent token-ms values; we
  // collapse them here to the phase-2-local flightFrac the math needs.
  // (Token phase3 = state-machine phase 2 — the dot-morph window.)
  const { flightMs, pulseMs } = tokens.motion.versionSwitch.phase3;
  const flightFrac = flightMs / (flightMs + pulseMs);

  const flightProgressFor = (_i: number): number => {
    if (phase < 2) return 0;
    if (phase === 2) return easeOutCubic(Math.min(progress / flightFrac, 1));
    return 1;
  };

  const pulseProgressFor = (i: number): number => {
    if (phase < 2) return 0;
    if (phase > 2) return 1;
    const pulseGlobal = (progress - flightFrac) / (1 - flightFrac);
    if (pulseGlobal <= 0) return 0;
    if (pulseGlobal >= 1) return 1;
    return waveProgress(pulseGlobal, delayNorm[i]!, spread, jitter, joined[i]!.id);
  };

  // Per-id schedule entries: absolute alpha (token-space, not multiplier),
  // and the lerped centrality scale * scalePop overlay. Lerping the
  // centrality scale through the same wave-pulse progress as color makes
  // each dot grow/shrink toward its new size *during* the wavefront —
  // not snap to a base scale at switch-start nor only after the entire
  // animation lands.
  const alphaById = new Map<number, number>();
  const radiusById = new Map<number, number>();
  const cParams = tokens.dot.centrality;
  for (let i = 0; i < joined.length; i++) {
    const j = joined[i]!;
    const p = pulseProgressFor(i);
    const fromAlpha = sideAlpha(j.fromCluster, j.toCluster);
    const toAlpha = sideAlpha(j.toCluster, j.fromCluster);
    alphaById.set(j.id, fromAlpha + (toAlpha - fromAlpha) * p);

    // Use the present-side centrality on both endpoints when one side
    // is missing — a fade-in/out dot keeps a meaningful scale.
    const fromC = j.fromCluster === null ? j.toCentrality : j.fromCentrality;
    const toC = j.toCluster === null ? j.fromCentrality : j.toCentrality;
    const fromCl = j.fromCluster ?? j.toCluster ?? -1;
    const toCl = j.toCluster ?? j.fromCluster ?? -1;
    const fromScale = centralityRadiusScale(fromCl, fromC, cParams);
    const toScale = centralityRadiusScale(toCl, toC, cParams);
    const lerpedScale = fromScale + (toScale - fromScale) * p;
    radiusById.set(j.id, lerpedScale * scalePop(p, scalePopPeak, scalePopUpFrac));
  }
  const users: DrawableUser[] = interpolateUsers(
    joined, flightProgressFor, pulseProgressFor,
    tokens.palette.cluster, tokens.palette.noise,
  )
    .map((u) => ({
      id: u.id, x: u.x, y: u.y, color: u.color,
      alpha: u.alpha * (alphaById.get(u.id) ?? tokens.dot.alpha),
      radiusScale: radiusById.get(u.id) ?? 1,
    }));
  return { users, alphaScale: 1, radiusScale: 1 };
};

export const DotsLayer = () => {
  const run = useStretchedRun();
  const transition = useTransition();
  const intro = useIntro();
  const [viewport] = useViewport();

  // joined and delayNorm depend only on the two runs in flight, not on
  // phase/progress — caching them prevents rebuilding the join + the
  // distance pass on every rAF tick.
  const joined = useMemo(
    () => transition ? joinUsersByCreator(transition.from, transition.to) : null,
    // eslint-disable-next-line react-hooks/exhaustive-deps -- joined depends only on the two runs in flight; the full transition object changes every rAF tick and would defeat the cache.
    [transition?.from, transition?.to],
  );
  const delayNorm = useMemo(
    () => joined ? computeWaveDelays(joined) : null,
    [joined],
  );

  const frame = useMemo<DotsFrame>(() => {
    if (transition && joined && delayNorm) {
      return morphFrame(joined, delayNorm, transition.phase, transition.progress);
    }
    if (intro) {
      return runToIntroDotsFrame(run, intro.centerWorld, intro.phase, intro.progress);
    }
    return runToDotsFrame(run);
  }, [run, transition, joined, delayNorm, intro]);

  const draw = useCallback(
    (g: PixiGraphics) => { drawDots(g, frame, viewport); },
    [frame, viewport],
  );

  return <pixiGraphics draw={draw} zIndex={1} />;
};
