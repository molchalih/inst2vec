import { Provider as JotaiProvider, useStore } from "jotai";
import { createContext, useContext, useMemo, type ReactNode } from "react";
import { makeSources, type ApiClient, type BulkSource } from "@/data";
import { setBulkSource, setApiClient, runStateAtom } from "@/state";
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
  return (
    <JotaiProvider>
      <SourceRegistration>{children}</SourceRegistration>
    </JotaiProvider>
  );
};

const SourceRegistration = ({ children }: { children: ReactNode }) => {
  const store = useStore();

  // One switch flips BOTH planes: `VITE_API_BASE_URL` set → HTTP sources,
  // unset → static JSON (the default; Pages stays unaffected). The active run
  // is read on every detail call so the API client follows version switches.
  const { bulk, api } = useMemo(() => {
    const getActiveRunId = () => store.get(runStateAtom).activeRunId;
    return makeSources(
      { baseUrl: config.baseUrl, apiBaseUrl: config.apiBaseUrl },
      getActiveRunId,
    );
  }, [store]);

  // Register module-level singletons consumed by atoms (kept React-free for
  // synchronous access).
  setBulkSource(bulk);
  setApiClient(api);

  return (
    <BulkContext.Provider value={bulk}>
      <ApiContext.Provider value={api}>{children}</ApiContext.Provider>
    </BulkContext.Provider>
  );
};
