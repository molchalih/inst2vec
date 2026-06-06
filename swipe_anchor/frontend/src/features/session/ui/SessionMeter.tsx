import { useAtomValue } from "jotai";
import { judgedCountAtom } from "@/state";

/**
 * Ambient session progress (plan §8.2 session). A quiet count of judgments made
 * this session — the seed of the engagement loop the live map will complete.
 */
export function SessionMeter() {
  const count = useAtomValue(judgedCountAtom);
  if (count === 0) return null;
  return (
    <div className="pointer-events-none absolute right-4 top-[max(10px,env(safe-area-inset-top))] z-40">
      <span className="rounded-chip border border-white/10 bg-bg-surface/70 px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted backdrop-blur">
        <span className="text-affirm">{count}</span> judged
      </span>
    </div>
  );
}
