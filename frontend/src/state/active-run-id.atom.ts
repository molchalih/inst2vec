import { atom } from "jotai";
import { caseAtom } from "./case.atom";
import { manifestAtom } from "./manifest.atom";

/**
 * Derived runId for the current case. Falls back to the manifest's
 * default_run_id if no run matches the active case (e.g. the exporter
 * has not yet produced that case). Null until the manifest loads.
 */
export const activeRunIdAtom = atom<string | null>((get) => {
  const manifest = get(manifestAtom);
  if (!manifest) return null;
  const active = get(caseAtom);
  const match = manifest.runs.find((r) => r.case === active);
  return match ? match.id : manifest.default_run_id;
});
