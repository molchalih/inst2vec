import { atom } from "jotai";
import { audioUnlockedAtom } from "@/state";

const STORAGE_KEY = "swipe-anchor.onboarded";

function initialOnboarded(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export const onboardedAtom = atom<boolean>(initialOnboarded());

/** Write-only: mark onboarding complete and persist it. */
export const completeOnboardingAtom = atom(null, (_get, set) => {
  try {
    localStorage.setItem(STORAGE_KEY, "1");
  } catch {
    /* private mode — in-memory only */
  }
  set(onboardedAtom, true);
  set(audioUnlockedAtom, true); // the Start tap is the gesture that lets reels play sound
});
