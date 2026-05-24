import { atom } from "jotai";
import type { Manifest } from "@/data";
import { requireBulkSource } from "./bulk-singleton";

export const manifestAtom = atom<Manifest | null>(null);

export const ensureManifestAtom = atom(null, async (get, set) => {
  if (get(manifestAtom)) return;
  const bulk = requireBulkSource();
  set(manifestAtom, await bulk.getManifest());
});
