/**
 * Ease-out cubic. The JS form of `tokens.motion.easeOut`. Used by Pixi-
 * side tweens (HoverLayer pulse, viewport refit, version-switch camera
 * and dot flight). Inputs outside [0, 1] are clamped so callers never
 * have to defend against rAF rounding.
 */
export const easeOutCubic = (t: number): number => {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  const u = 1 - t;
  return 1 - u * u * u;
};
