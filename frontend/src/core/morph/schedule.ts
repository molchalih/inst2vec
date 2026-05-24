import type { TransitionPhase } from "./types";

export const ellipseAlphaScale = (
  phase: TransitionPhase,
  progress: number,
): number => {
  switch (phase) {
    case 0: return 1;
    case 1: return 1 - progress;
    case 2: return 0;
    case 3: return progress;
  }
};

export const ellipseSide = (phase: TransitionPhase): "from" | "to" =>
  phase < 2 ? "from" : "to";

export const userAlphaSchedule = (input: {
  inFrom: boolean;
  inTo: boolean;
  progress: number;
}): number => {
  if (input.inFrom && input.inTo) return 1;
  if (input.inFrom) return 1 - input.progress;
  if (input.inTo) return input.progress;
  return 0;
};
