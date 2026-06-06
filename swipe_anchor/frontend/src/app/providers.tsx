import type { ReactNode } from "react";
import { Provider as JotaiProvider } from "jotai";
import { HttpApiClient } from "@/data";
import { getAccessCode, setApiClient } from "@/state";
import { config } from "./config";

// Register the ApiClient once at module load via the state singleton, so async
// action atoms can reach it without React context (mirrors the atlas pattern).
// The per-user deeplink code is attached as the X-Access-Code header (auth); no
// build-time bearer secret is embedded (see app/config.ts).
setApiClient(new HttpApiClient(config.apiBaseUrl, getAccessCode));

export function Providers({ children }: { children: ReactNode }) {
  return <JotaiProvider>{children}</JotaiProvider>;
}
