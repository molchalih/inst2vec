import { atom, useAtomValue } from "jotai";

/**
 * Whether the user has interacted enough for the browser to allow audio. Reels
 * start muted (autoplay policy); the first tap/swipe/Start flips this and the
 * active reel plays with its own soundtrack.
 */
export const audioUnlockedAtom = atom<boolean>(false);

export const useAudioUnlocked = () => useAtomValue(audioUnlockedAtom);
