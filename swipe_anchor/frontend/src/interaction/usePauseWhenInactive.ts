import { useEffect, useRef } from "react";

/**
 * Reel playback control (plan §8.3): keep at most one `<video>` decoding. The
 * front card passes `active=true` to play; others pause + rewind so mobile
 * memory/battery isn't spent off-screen. `muted` gates the reel's own audio —
 * it stays muted until the user has interacted (browser autoplay policy), then
 * the active reel plays with sound.
 */
export function usePauseWhenInactive(active: boolean, muted = true) {
  const ref = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.muted = muted;
    if (active) {
      void el.play?.().catch(() => {});
    } else {
      // Pause but KEEP the current frame (no currentTime reset) so swiping back
      // to this card shows imagery immediately instead of a black flash.
      el.pause?.();
    }
  }, [active, muted]);

  return ref;
}
