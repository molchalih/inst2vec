import { useEffect, useState } from "react";
import { useAtomValue, useSetAtom, useStore } from "jotai";
import { accessCodeAtom, batchQueueAtom, ensureBatchAtom } from "@/state";
import {
  AccessRequired,
  JudgeScreen,
  OnboardingSheet,
  onboardedAtom,
} from "@/features";
import { isTelegram } from "@/data/telegram";
import { config } from "./config";
import { runAccessBootstrap } from "./access-bootstrap";
import { Providers } from "./providers";

/** Headless: warm the comparison queue once on mount (prefetch is fine during
 *  onboarding — only the judge view's timers must wait). */
function BatchLoader() {
  const ensureBatch = useSetAtom(ensureBatchAtom);
  useEffect(() => {
    void ensureBatch();
  }, [ensureBatch]);
  return null;
}

/**
 * Warm the reels for the NEXT comparison so a commit never waits on a download.
 * The current comparison's reels are loaded by the mounted cards themselves, so
 * we skip them here to avoid duplicate fetches — except during onboarding, when
 * the judge view isn't mounted yet, where we also warm the first set.
 */
function MediaPrefetch() {
  const queue = useAtomValue(batchQueueAtom);
  const onboarded = useAtomValue(onboardedAtom);
  const items = onboarded ? queue.slice(1, 2) : queue.slice(0, 2);
  const urls = items
    .flatMap((item) => item.creators.map((c) => c.clips[0]?.video_url))
    .filter((u): u is string => Boolean(u));
  return (
    <div aria-hidden className="pointer-events-none absolute h-0 w-0 overflow-hidden opacity-0">
      {urls.map((u) => (
        <video key={u} src={u} preload="auto" muted playsInline />
      ))}
    </div>
  );
}

/**
 * Mount the judge loop only AFTER onboarding is accepted, so the reaction- and
 * dwell-timers never start against a comparison hidden behind the onboarding
 * sheet (which would fold reading time into the first judgment's metrics).
 */
function JudgeGate() {
  const onboarded = useAtomValue(onboardedAtom);
  return onboarded ? <JudgeScreen /> : null;
}

/**
 * Gate the entire app on a deeplink access code. With no code (URL or storage)
 * nothing opens — not even the onboarding sheet or the batch prefetch.
 */
/**
 * Resolving spinner shown only while the Telegram bootstrap is exchanging
 * initData for a code, so the AccessRequired gate never flashes for a valid
 * Telegram user. Non-Telegram launches never enter this state.
 */
function Resolving() {
  return (
    <main className="relative mx-auto flex h-[100dvh] w-full max-w-md items-center justify-center overflow-hidden bg-bg-canvas">
      <p className="font-mono text-[10px] uppercase tracking-[0.3em] text-fg-faint">
        opening…
      </p>
    </main>
  );
}

function AppBody() {
  const store = useStore();
  const code = useAtomValue(accessCodeAtom);
  // Only Telegram launches need an async resolve; the browser path is already
  // resolved synchronously at module load, so it never shows the spinner.
  const [resolving, setResolving] = useState(() => !code && isTelegram());

  useEffect(() => {
    if (!resolving) return;
    let active = true;
    void runAccessBootstrap(store, config.apiBaseUrl).finally(() => {
      if (active) setResolving(false);
    });
    return () => {
      active = false;
    };
  }, [resolving, store]);

  if (resolving) return <Resolving />;

  if (!code) {
    return (
      <main className="relative mx-auto h-[100dvh] w-full max-w-md overflow-hidden bg-bg-canvas">
        <AccessRequired />
      </main>
    );
  }
  return (
    <>
      <BatchLoader />
      <MediaPrefetch />
      <main className="relative mx-auto h-[100dvh] w-full max-w-md overflow-hidden bg-bg-canvas">
        {/* judged-count meter hidden from the UI on purpose; the count still
            tracks in judgedCountAtom and every judgment is recorded server-side. */}
        <JudgeGate />
        <OnboardingSheet />
      </main>
    </>
  );
}

export function AppShell() {
  return (
    <Providers>
      <AppBody />
    </Providers>
  );
}
