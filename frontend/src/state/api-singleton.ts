import type { ApiClient } from "@/data";

/**
 * Module-level singleton injected by app/providers.tsx at startup.
 * Atoms read it via requireApiClient() so they stay React-free.
 */
let instance: ApiClient | null = null;

export const setApiClient = (c: ApiClient): void => { instance = c; };

export const requireApiClient = (): ApiClient => {
  if (!instance) throw new Error("ApiClient not provided");
  return instance;
};
