import { easeOutCubic } from "../motion/ease";
import { lerp } from "./interpolate";
import type { JoinedUser } from "./join";

/**
 * Phase-2-local eased flight progress for the uniform position morph. Mirrors
 * the per-frame `flightProgressFor` the dots layer applies: frozen at 0 before
 * the flight window (phases 0/1), eased `progress / flightFrac` (clamped to 1)
 * during phase 2, and held at 1 afterwards (phase 3). Extracted to core so the
 * dots layer and the tracking overlay share one source of motion truth and the
 * halo sits exactly on the morphing dot, frame for frame.
 */
export const flightProgress = (
  phase: 0 | 1 | 2 | 3,
  progress: number,
  flightFrac: number,
): number => {
  if (phase < 2) return 0;
  if (phase === 2) return easeOutCubic(Math.min(progress / flightFrac, 1));
  return 1;
};

/**
 * Mid-flight world position of a single creator during a version-switch morph.
 * Finds the joined row by creator id and returns its lerped `[x, y]` at the
 * given motion progress: a plain both-sides lerp when present on both runs, the
 * present side's position when present on only one, and `null` when the creator
 * is absent from the join. Pure position branch of `interpolateUsers`, shared so
 * the tracking halo travels on exactly the same point the base dot is drawn at.
 */
export const interpolatedUserPos = (
  joined: ReadonlyArray<JoinedUser>,
  creatorId: number,
  motionProgress: number,
): [number, number] | null => {
  const j = joined.find((u) => u.id === creatorId);
  if (!j) return null;
  const { fromXY, toXY } = j;
  if (fromXY && toXY) {
    return [
      lerp(fromXY[0], toXY[0], motionProgress),
      lerp(fromXY[1], toXY[1], motionProgress),
    ];
  }
  if (fromXY) return [fromXY[0], fromXY[1]];
  if (toXY) return [toXY[0], toXY[1]];
  return null;
};
