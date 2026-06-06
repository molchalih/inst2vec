import { atom, useAtom } from "jotai";

/**
 * Count of judgments submitted this session — drives the engagement meter. The
 * annotator identity itself is the deeplink access code (see access.atom.ts);
 * the backend derives it from the `X-Access-Code` header, so the client no longer
 * mints an anonymous id.
 */
export const judgedCountAtom = atom<number>(0);

export const useJudgedCount = () => useAtom(judgedCountAtom);
