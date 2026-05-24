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

  // selection → route: write cluster= or user= into route.
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

  // route → URL hash.
  useEffect(() => {
    const next = serializeRoute(route);
    if (window.location.hash !== next) {
      const url = window.location.pathname + window.location.search + next;
      window.history.replaceState(null, "", url);
    }
  }, [route]);

  // URL hashchange → route + selection.
  useEffect(() => {
    const onHashChange = (): void => {
      const parsed = parseHash(window.location.hash);
      setRoute(parsed);
      const next: Selection =
        parsed.user !== undefined
          ? { kind: "creator", creatorId: parsed.user }
          : parsed.cluster !== undefined
            ? { kind: "cluster", clusterId: parsed.cluster }
            : null;
      setSelection(next);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [setRoute, setSelection]);
};
