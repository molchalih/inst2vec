import { atom } from "jotai";
import type { EmbeddingCase } from "@/data";
import { routeAtom } from "./route.atom";

const DEFAULT_CASE: EmbeddingCase = "video";

export const caseAtom = atom<EmbeddingCase>(
  (get) => get(routeAtom).case ?? DEFAULT_CASE,
);
