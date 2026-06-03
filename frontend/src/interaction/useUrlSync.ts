import { useEffect } from "react";
import { useAtom } from "jotai";
import { routeAtom, parseHash, serializeRoute, selectionAtom } from "@/state";
import type { Selection } from "@/state";

/**
 * Bidirectional sync between routeAtom/selectionAtom and window.location.hash.
 * - On mount: hydrate atoms from current hash.
 * - On route change: write a new hash (no history entry).
 * - On selection change: update route's cluster/user keys.
 * - On hashchange event: re-hydrate atoms.
 */
export const useUrlSync = (): void => {
  const [route, setRoute] = useAtom(routeAtom);
  const [selection, setSelection] = useAtom(selectionAtom);

  useEffect(() => {
    if (selection === null) {
      setRoute((r) => {
        const next = { ...r };
        delete next.cluster;
        delete next.user;
        return next;
      });
    } else if (selection.kind === "cluster") {
      setRoute((r) => ({ ...r, cluster: selection.clusterId, user: undefined }));
    } else {
      setRoute((r) => ({ ...r, user: selection.creatorId, cluster: undefined }));
    }
  }, [selection, setRoute]);

  useEffect(() => {
    const next = serializeRoute(route);
    if (globalThis.location.hash === next) return;
    const url = globalThis.location.pathname + globalThis.location.search + next;
    globalThis.history.replaceState(null, "", url);
  }, [route]);

  useEffect(() => {
    const onHashChange = (): void => {
      const parsed = parseHash(globalThis.location.hash);
      setRoute(parsed);
      let next: Selection = null;
      if (parsed.user !== undefined) {
        next = { kind: "creator", creatorId: parsed.user };
      } else if (parsed.cluster !== undefined) {
        next = { kind: "cluster", clusterId: parsed.cluster };
      }
      setSelection(next);
    };
    globalThis.addEventListener("hashchange", onHashChange);
    return () => globalThis.removeEventListener("hashchange", onHashChange);
  }, [setRoute, setSelection]);
};
