import { afterEach, describe, expect, it, vi } from "vitest";
import { createStore } from "jotai";
import { accessCodeAtom, getAccessCode } from "@/state";
import { runAccessBootstrap } from "./access-bootstrap";

type WebApp = { initData: string; ready: () => void; expand: () => void };

function installTelegram(initData: string): void {
  (window as unknown as { Telegram?: { WebApp: WebApp } }).Telegram = {
    WebApp: { initData, ready: vi.fn(), expand: vi.fn() },
  };
}

afterEach(() => {
  delete (window as unknown as { Telegram?: unknown }).Telegram;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("runAccessBootstrap", () => {
  it("does nothing outside Telegram (keeps the ?code= path untouched)", async () => {
    const store = createStore();
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await runAccessBootstrap(store, "http://api.test");

    expect(fetchMock).not.toHaveBeenCalled();
    expect(store.get(accessCodeAtom)).toBeNull();
  });

  it("sets the code from /tg/auth when in Telegram with no stored code", async () => {
    installTelegram("auth_date=1&hash=abc");
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ access_code: "tg7" }), { status: 200 }),
      ),
    );
    const store = createStore();

    await runAccessBootstrap(store, "http://api.test");

    expect(store.get(accessCodeAtom)).toBe("tg7");
    expect(getAccessCode()).toBe("tg7");
  });

  it("leaves the code null when /tg/auth rejects (unregistered)", async () => {
    installTelegram("auth_date=1&hash=abc");
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("no", { status: 403 })),
    );
    const store = createStore();

    await runAccessBootstrap(store, "http://api.test");

    expect(store.get(accessCodeAtom)).toBeNull();
  });
});
