import type { BulkSource } from "@/data";

/**
 * Module-level singleton injected by app/providers.tsx at startup.
 * Atoms stay synchronous and React-free; the bulk source is the only
 * side-effect channel they touch.
 */
let instance: BulkSource | null = null;

export const setBulkSource = (s: BulkSource): void => { instance = s; };

export const requireBulkSource = (): BulkSource => {
  if (!instance) throw new Error("BulkSource not provided");
  return instance;
};
