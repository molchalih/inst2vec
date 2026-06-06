import { atom, useAtomValue } from "jotai";

const STORAGE_KEY = "swipe-anchor.access-code";

/**
 * Resolve the deeplink access code (auth/identity). Precedence: a `?code=` query
 * param (persisted, then stripped from the URL so it isn't shared/bookmarked in
 * the clear) → previously stored code → null. Runs once at module load.
 */
function resolveAccessCode(): string | null {
  try {
    const url = new URL(window.location.href);
    const fromUrl = url.searchParams.get("code");
    if (fromUrl && fromUrl.trim()) {
      const code = fromUrl.trim();
      localStorage.setItem(STORAGE_KEY, code);
      url.searchParams.delete("code");
      window.history.replaceState({}, "", url.toString());
      return code;
    }
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

let currentCode: string | null = resolveAccessCode();

/** Read by the ApiClient to attach the `X-Access-Code` header (DI accessor). */
export function getAccessCode(): string | null {
  return currentCode;
}

/** Forget the stored code (e.g. an explicit sign-out). */
export function clearAccessCode(): void {
  currentCode = null;
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

/**
 * Set + persist the access code at runtime (used by the Telegram bootstrap once
 * it exchanges initData for a code). Updates the module var the ApiClient reads
 * and persists to storage; callers update `accessCodeAtom` via a jotai store so
 * the gate re-renders.
 */
export function setAccessCode(code: string): void {
  const trimmed = code.trim();
  if (!trimmed) return;
  currentCode = trimmed;
  try {
    localStorage.setItem(STORAGE_KEY, trimmed);
  } catch {
    /* ignore */
  }
}

export const accessCodeAtom = atom<string | null>(currentCode);

export const useAccessCode = () => useAtomValue(accessCodeAtom);
