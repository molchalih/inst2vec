import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getInitData,
  isTelegram,
  exchangeInitDataForCode,
} from "./telegram";

type WebApp = {
  initData: string;
  ready: () => void;
  expand: () => void;
};

function installTelegram(initData: string): { ready: ReturnType<typeof vi.fn>; expand: ReturnType<typeof vi.fn> } {
  const ready = vi.fn();
  const expand = vi.fn();
  const webApp: WebApp = { initData, ready, expand };
  (window as unknown as { Telegram?: { WebApp: WebApp } }).Telegram = {
    WebApp: webApp,
  };
  return { ready, expand };
}

afterEach(() => {
  delete (window as unknown as { Telegram?: unknown }).Telegram;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("isTelegram / getInitData", () => {
  it("is false with no Telegram global", () => {
    expect(isTelegram()).toBe(false);
    expect(getInitData()).toBe("");
  });

  it("is true and returns initData when the WebApp is present", () => {
    installTelegram("auth_date=1&hash=abc");
    expect(isTelegram()).toBe(true);
    expect(getInitData()).toBe("auth_date=1&hash=abc");
  });

  it("is false when Telegram exists but initData is empty", () => {
    installTelegram("");
    expect(isTelegram()).toBe(false);
  });
});

describe("exchangeInitDataForCode", () => {
  it("returns the access code on a 200 from /tg/auth", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ access_code: "tg42" }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const code = await exchangeInitDataForCode(
      "http://api.test",
      "auth_date=1&hash=abc",
    );
    expect(code).toBe("tg42");
    expect(fetchMock).toHaveBeenCalledWith(
      "http://api.test/tg/auth",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("returns null on a non-200 (e.g. 403 unregistered)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("nope", { status: 403 })),
    );
    expect(
      await exchangeInitDataForCode("http://api.test", "x"),
    ).toBeNull();
  });

  it("returns null when fetch throws (offline)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("offline");
      }),
    );
    expect(
      await exchangeInitDataForCode("http://api.test", "x"),
    ).toBeNull();
  });
});
