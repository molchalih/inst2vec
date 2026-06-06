/**
 * Thin, swappable Telegram Mini App adapter. Everything Telegram-specific lives
 * here so the rest of the app keeps depending only on the access-code identity.
 * The non-Telegram (`?code=`) path is untouched: these helpers are no-ops unless
 * the Telegram WebApp SDK is present with non-empty initData.
 */

interface TelegramWebApp {
  initData: string;
  ready: () => void;
  expand: () => void;
}

function webApp(): TelegramWebApp | null {
  const tg = (window as unknown as { Telegram?: { WebApp?: TelegramWebApp } })
    .Telegram;
  return tg?.WebApp ?? null;
}

/** True only when launched inside Telegram with signed initData to validate. */
export function isTelegram(): boolean {
  const app = webApp();
  return Boolean(app && app.initData && app.initData.length > 0);
}

/** The raw signed initData query string (empty string when not in Telegram). */
export function getInitData(): string {
  return webApp()?.initData ?? "";
}

/** Tell Telegram we're loaded and want full height. Safe to call when absent. */
export function initTelegram(): void {
  const app = webApp();
  if (!app) return;
  try {
    app.ready();
    app.expand();
  } catch {
    /* ignore — older clients may lack these */
  }
}

/**
 * Exchange signed initData for an access code via the backend. Returns the code
 * on success, or `null` on any failure (bad signature 401, unregistered 403,
 * or network error) — the caller then falls back to the normal gate.
 */
export async function exchangeInitDataForCode(
  apiBaseUrl: string,
  initData: string,
): Promise<string | null> {
  try {
    const res = await fetch(`${apiBaseUrl}/tg/auth`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ init_data: initData }),
    });
    if (!res.ok) return null;
    const body = (await res.json()) as { access_code?: string };
    return body.access_code ?? null;
  } catch {
    return null;
  }
}

/**
 * Ask the backend to DM this Telegram user a "come back" nudge later (the bot
 * sends it ~20h out). Authenticated by the same signed initData. Returns true on
 * success; a no-op returning false outside Telegram or on any error.
 */
export async function remindLater(apiBaseUrl: string): Promise<boolean> {
  const initData = getInitData();
  if (!initData) return false;
  try {
    const res = await fetch(`${apiBaseUrl}/tg/remind`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ init_data: initData }),
    });
    return res.ok;
  } catch {
    return false;
  }
}
