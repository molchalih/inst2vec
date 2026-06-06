import type { ApiClient } from "@/data";

/**
 * Module-level ApiClient registrar (mirrors the atlas `state/api-singleton`
 * carve-out): lets async action atoms reach the network without React context,
 * keeping the atoms themselves free of provider plumbing. `app/` registers the
 * client once at startup.
 */
let client: ApiClient | null = null;

export function setApiClient(c: ApiClient): void {
  client = c;
}

export function requireApiClient(): ApiClient {
  if (!client) throw new Error("ApiClient not registered — call setApiClient in app/");
  return client;
}
