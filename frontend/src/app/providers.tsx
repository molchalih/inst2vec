import { Provider as JotaiProvider, useStore } from "jotai";
import { createContext, useContext, useMemo, type ReactNode } from "react";
import {
  HttpApiClient, StaticApiClient, StaticBulkSource,
  type ApiClient, type BulkSource,
} from "@/data";
import { setBulkSource, setApiClient } from "@/state";
import { runStateAtom } from "@/state";
import { config } from "./config";

const BulkContext = createContext<BulkSource | null>(null);
const ApiContext = createContext<ApiClient | null>(null);

export const useBulk = (): BulkSource => {
  const v = useContext(BulkContext);
  if (!v) throw new Error("BulkProvider missing");
  return v;
};

export const useApi = (): ApiClient => {
  const v = useContext(ApiContext);
  if (!v) throw new Error("ApiProvider missing");
  return v;
};

type ProvidersProps = { children: ReactNode };

export const Providers = ({ children }: ProvidersProps) => {
  const bulk = useMemo<BulkSource>(
    () => new StaticBulkSource(`${config.baseUrl}data/`),
    [],
  );

  // Register the bulk source as the module-level singleton consumed
  // by ensureRunAtom. Doing this at provider construction keeps atoms
  // synchronous and free of React context.
  setBulkSource(bulk);

  return (
    <JotaiProvider>
      <ApiRegistration bulk={bulk}>{children}</ApiRegistration>
    </JotaiProvider>
  );
};

const ApiRegistration = ({ bulk, children }: { bulk: BulkSource; children: ReactNode }) => {
  const store = useStore();
  const api = useMemo<ApiClient>(() => {
    const getActiveRunId = () => store.get(runStateAtom).activeRunId;
    return config.apiBaseUrl
      ? new HttpApiClient(config.apiBaseUrl)
      : new StaticApiClient(`${config.baseUrl}data/`, getActiveRunId);
  }, [store]);

  // Register singleton consumed by atoms (kept React-free for sync access).
  setApiClient(api);

  return (
    <BulkContext.Provider value={bulk}>
      <ApiContext.Provider value={api}>{children}</ApiContext.Provider>
    </BulkContext.Provider>
  );
};
