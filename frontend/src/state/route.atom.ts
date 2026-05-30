import { atom, useAtom } from "jotai";
import { routeSchema, type Route } from "./route.schema";

const parseHash = (hash: string): Route => {
  const trimmed = hash.startsWith("#") ? hash.slice(1) : hash;
  if (!trimmed) return {};
  const params = new URLSearchParams(trimmed);
  const obj: Record<string, string> = {};
  for (const [k, v] of params) obj[k] = v;
  const parsed = routeSchema.safeParse(obj);
  return parsed.success ? parsed.data : {};
};

const serializeRoute = (r: Route): string => {
  const params = new URLSearchParams();
  if (r.case) params.set("case", r.case);
  if (r.cluster !== undefined) params.set("cluster", String(r.cluster));
  if (r.user !== undefined) params.set("user", String(r.user));
  const s = params.toString();
  return s ? `#${s}` : "";
};

export const routeAtom = atom<Route>(
  typeof globalThis === "undefined" ? {} : parseHash(globalThis.location.hash),
);

export { parseHash, serializeRoute };
export const useRoute = () => useAtom(routeAtom);
