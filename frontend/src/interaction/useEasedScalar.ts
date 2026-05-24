import { useTween } from "./useTween";

export const useEasedScalar = (
  active: boolean,
  duration: number,
  ease: (t: number) => number,
): number => useTween(active ? 1 : 0, duration, ease);
