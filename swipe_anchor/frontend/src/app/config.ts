/**
 * Typed runtime config. Only the API base url is configurable from the browser
 * (`VITE_API_BASE_URL`); the dev default points at the local FastAPI backend
 * (`python -m swipe_anchor.backend` on :8100).
 *
 * No bearer token is read here ON PURPOSE: Vite inlines `VITE_*` values into the
 * public bundle, so any embedded secret is trivially extractable and provides no
 * real protection (plan §7.5). The browser app runs in **public mode** against a
 * no-token backend; a token-protected deployment must sit behind a server-side
 * proxy / session that injects auth, never the client bundle.
 */
interface AppConfig {
  apiBaseUrl: string;
}

const env = import.meta.env as Record<string, string | undefined>;

export const config: AppConfig = {
  apiBaseUrl: env.VITE_API_BASE_URL ?? "http://localhost:8100",
};
