import type { createStore } from "jotai";
import { accessCodeAtom, getAccessCode, setAccessCode } from "@/state";
import {
  exchangeInitDataForCode,
  getInitData,
  initTelegram,
  isTelegram,
} from "@/data/telegram";

type Store = ReturnType<typeof createStore>;

/**
 * One-shot access bootstrap. Non-Telegram launches are a no-op (the synchronous
 * `?code=` resolution already ran at module load). Inside Telegram with no
 * stored code, exchange initData for an access code and publish it to the atom
 * so the gate resolves without a flash.
 */
export async function runAccessBootstrap(
  store: Store,
  apiBaseUrl: string,
): Promise<void> {
  if (!isTelegram()) return;
  initTelegram();
  if (getAccessCode()) return; // a returning Telegram user already has a code
  const code = await exchangeInitDataForCode(apiBaseUrl, getInitData());
  if (code) {
    setAccessCode(code);
    store.set(accessCodeAtom, code);
  }
}
