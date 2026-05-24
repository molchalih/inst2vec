/**
 * Deterministic per-id hash in [0, 1). A finalizer-style integer mix
 * (the same one previously inlined in core/morph/wave.ts). Used for
 * jitter/stagger that must be stable across frames and reproducible
 * per dot id.
 */
export const hashUnit = (id: number): number => {
  let x = (id ^ 0x9e3779b9) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0;
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x / 0xffffffff;
};
