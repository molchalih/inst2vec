import { atom } from "jotai";

/**
 * Drawer open/closed state. Feature-local: not in src/state/ because
 * it is single-consumer, view-only, and never URL-synced. Closed by
 * default; the chevron-tongue is the only thing visible.
 */
export const drawerOpenAtom = atom<boolean>(false);
