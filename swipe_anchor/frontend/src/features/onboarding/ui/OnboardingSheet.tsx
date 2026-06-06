import { useAtomValue, useSetAtom } from "jotai";
import { tokens } from "@/ui";
import { completeOnboardingAtom, onboardedAtom } from "../state.atom";

/**
 * First-session consent + worked example (plan §8.1 onboarding). Renders nothing
 * once accepted. The worked example doubles as gold calibration: an obvious odd
 * one (a chef among two DJs).
 */
export function OnboardingSheet() {
  const onboarded = useAtomValue(onboardedAtom);
  const complete = useSetAtom(completeOnboardingAtom);

  if (onboarded) return null;

  return (
    <div className="absolute inset-0 z-50 flex flex-col bg-bg-canvas/96 backdrop-blur">
      <div className="triage-scroll flex flex-1 flex-col justify-center gap-8 px-7">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.3em] text-accent">
            triage
          </p>
          <h1 className="mt-2 font-display text-[34px] font-extrabold leading-[1.05] text-fg-default">
            Which creator
            <br />
            doesn&apos;t belong?
          </h1>
          <p className="mt-4 max-w-sm font-mono text-[12px] leading-relaxed text-fg-muted">
            Swipe ← → through three creators. <span className="text-fg-default">Tap</span>{" "}
            the one that fits least to cross it out (tap again to undo), then{" "}
            <span className="text-fg-default">swipe up</span> to send. If they all feel
            alike, hit <span className="text-fg-default">Skip</span>. No single right
            answer — go with your gut.
          </p>
        </div>

        <Example />

        <div className="max-w-sm rounded-lg border-l-2 border-accent bg-accent/10 px-3 py-2.5">
          <p className="font-mono text-[12px] leading-relaxed text-fg-default">
            Can&apos;t see a real difference? Hit{" "}
            <span className="text-accent">Skip</span>. &quot;Too close to tell&quot; is a
            real, useful answer — don&apos;t force a pick.
          </p>
        </div>

        <p className="max-w-sm font-mono text-[10px] leading-relaxed text-fg-faint">
          Anonymous — no login, no personal data stored. Your judgments train a fairer
          similarity map.
        </p>
      </div>

      <div className="px-7 pb-[max(20px,env(safe-area-inset-bottom))]">
        <button
          type="button"
          onClick={() => complete()}
          className="h-14 w-full rounded-bar bg-accent font-display text-[16px] font-semibold uppercase tracking-wide text-bg-canvas transition duration-fast ease-out active:translate-y-0.5"
        >
          Start judging
        </button>
      </div>
    </div>
  );
}

function Example() {
  const items = [
    { label: "DJ", odd: false },
    { label: "DJ", odd: false },
    { label: "Chef", odd: true },
  ];
  return (
    <div className="flex gap-3">
      {items.map((it, i) => (
        <div
          key={i}
          className="relative flex h-24 flex-1 items-center justify-center rounded-card border border-white/10 bg-bg-raised font-display text-lg text-fg-default"
        >
          {it.label}
          {it.odd && (
            <span
              className="absolute left-[-6%] right-[-6%] h-[3px] bg-accent"
              style={{ transform: `rotate(${tokens.card.crossSlashDeg}deg)` }}
            />
          )}
        </div>
      ))}
    </div>
  );
}
