import { useEffect, useRef, useState } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  audioUnlockedAtom,
  batchStatusAtom,
  crossedAtom,
  currentItemAtom,
  ensureBatchAtom,
  submitErrorAtom,
  submitJudgmentAtom,
} from "@/state";
import { useDwellTimer, useReactionTimer, useSwipe } from "@/interaction";
import { ActionButton, Pager, tokens } from "@/ui";
import { config } from "@/app/config";
import { isTelegram, remindLater } from "@/data/telegram";
import { CreatorCard } from "./CreatorCard";
import { detectTired, median } from "../coaching";

const LABELS = ["A", "B", "C"];

// Coaching thresholds.
const NO_BROWSE_STREAK = 3; // commits in a row without ever swiping to compare
const REST_SAMPLES = 5; // how many recent answers to judge "rushing" over
const REST_MEDIAN_MS = 2500; // median answer time below this = too fast

type Coach = "swipe" | "rest" | "tired" | null;

export function JudgeScreen() {
  const item = useAtomValue(currentItemAtom);
  const status = useAtomValue(batchStatusAtom);
  const crossed = useAtomValue(crossedAtom);
  const setCrossed = useSetAtom(crossedAtom);
  const submit = useSetAtom(submitJudgmentAtom);
  const ensureBatch = useSetAtom(ensureBatchAtom);
  const submitError = useAtomValue(submitErrorAtom);
  const audioUnlocked = useAtomValue(audioUnlockedAtom);
  const unlockAudio = useSetAtom(audioUnlockedAtom);

  const [activeIndex, setActiveIndex] = useState(0);
  const [pulseKey, setPulseKey] = useState(0);
  const [coach, setCoach] = useState<Coach>(null);

  // Coaching signals (refs so they don't trigger renders).
  const browsed = useRef(false); // did they swipe to compare on this case?
  const noBrowseStreak = useRef(0);
  const rtHistory = useRef<number[]>([]);

  const comparisonId = item?.comparison_id ?? null;
  const activeCreatorId = item?.creators[activeIndex]?.creator_id ?? null;
  const readReaction = useReactionTimer(comparisonId);
  const readDwell = useDwellTimer(activeCreatorId, comparisonId);

  const send = (oddId: number | null) => {
    const rt = readReaction();
    void submit({
      oddId,
      confidence: 1,
      reactionMs: rt,
      dwell: readDwell(),
      expanded: false,
    }).catch(() => {});
    setPulseKey((k) => k + 1);

    // --- coaching ---
    noBrowseStreak.current = browsed.current ? 0 : noBrowseStreak.current + 1;
    if (rt != null) {
      rtHistory.current.push(rt);
      if (rtHistory.current.length > 8) rtHistory.current.shift();
    }
    const recent = rtHistory.current.slice(-REST_SAMPLES);
    if (detectTired(rtHistory.current)) {
      setCoach("tired"); // slowing down / labouring → offer a break + reminder
      rtHistory.current = [];
    } else if (recent.length >= REST_SAMPLES && median(recent) < REST_MEDIAN_MS) {
      setCoach("rest");
      rtHistory.current = []; // don't nag again until a fresh fast streak builds
    } else if (noBrowseStreak.current >= NO_BROWSE_STREAK) {
      setCoach("swipe");
      noBrowseStreak.current = 0;
    }
  };

  const onSwipe = (dir: "left" | "right" | "up" | "down") => {
    unlockAudio(true);
    if (dir === "left") {
      setActiveIndex((i) => {
        const next = Math.min(2, i + 1); // browse forward
        if (next !== i) browsed.current = true; // only count an actual move
        return next;
      });
    } else if (dir === "right") {
      setActiveIndex((i) => {
        const next = Math.max(0, i - 1); // browse back
        if (next !== i) browsed.current = true;
        return next;
      });
    } else if (dir === "up" && crossed !== null) {
      send(crossed); // commit the marked odd one → new case
    }
  };

  const onTap = () => {
    if (activeCreatorId === null) return;
    unlockAudio(true);
    setCrossed((c) => (c === activeCreatorId ? null : activeCreatorId));
  };

  const { dragX, dragging, bind } = useSwipe({ onSwipe, onTap });

  useEffect(() => {
    setActiveIndex(0);
    browsed.current = false; // reset browse tracking for the new comparison
  }, [comparisonId]);

  // Auto-recover from a transient fetch failure (cold tunnel / QUIC blip) so the
  // user isn't stranded on "couldn't reach the server".
  const retries = useRef(0);
  useEffect(() => {
    if (status !== "error") {
      retries.current = 0;
      return;
    }
    if (retries.current >= 5) return;
    const t = setTimeout(() => {
      retries.current += 1;
      void ensureBatch();
    }, 2000);
    return () => clearTimeout(t);
  }, [status, ensureBatch]);

  if (!item) {
    return <StatusScreen status={status} onRetry={() => void ensureBatch()} />;
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between px-5 pt-[max(10px,env(safe-area-inset-top))] pb-2">
        <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-fg-faint">
          which one doesn&apos;t belong?
        </span>
        <Pager count={3} index={activeIndex} />
      </header>

      <div className="relative flex-1 touch-none overflow-hidden px-3 pb-3" {...bind}>
        <div
          className="flex h-full"
          style={{
            transform: `translateX(calc(${-activeIndex * 100}% + ${dragX}px))`,
            transition: dragging
              ? "none"
              : `transform ${tokens.motion.swipe}ms ${tokens.motion.easeGlide}`,
          }}
        >
          {item.creators.map((card, i) => (
            <div key={card.creator_id} className="h-full w-full shrink-0 px-1">
              <CreatorCard
                card={card}
                label={LABELS[i] ?? String(i + 1)}
                isOdd={crossed === card.creator_id}
                active={i === activeIndex}
                audioUnlocked={audioUnlocked}
                onToggleOdd={() => {
                  unlockAudio(true);
                  setCrossed(crossed === card.creator_id ? null : card.creator_id);
                }}
                onSkip={() => {
                  unlockAudio(true);
                  send(null);
                }}
              />
            </div>
          ))}
        </div>

        {pulseKey > 0 && (
          <span
            key={pulseKey}
            className="pointer-events-none absolute left-1/2 top-1/2 h-24 w-24 -translate-x-1/2 -translate-y-1/2 rounded-full"
            style={{
              border: `2px solid ${tokens.affirm.base}`,
              animation: `triage-pulse ${tokens.motion.slow}ms ${tokens.motion.easeOut} forwards`,
            }}
          />
        )}

        {coach && <CoachOverlay variant={coach} onDismiss={() => setCoach(null)} />}
      </div>

      <div className="flex h-9 items-center justify-center px-5">
        {submitError ? (
          <p className="font-mono text-[10px] uppercase tracking-[0.14em] text-accent">
            couldn&apos;t save — this card came back, try again
          </p>
        ) : crossed !== null ? (
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-affirm">
            swipe up to continue ↑
          </p>
        ) : (
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-fg-faint">
            ← swipe to compare · tap the odd one
          </p>
        )}
      </div>
    </div>
  );
}

function CoachOverlay({
  variant,
  onDismiss,
}: {
  variant: "swipe" | "rest" | "tired";
  onDismiss: () => void;
}) {
  if (variant === "tired") {
    return <RestNudge onDismiss={onDismiss} />;
  }
  const copy =
    variant === "swipe"
      ? {
          kicker: "tip",
          title: "Compare before you choose",
          body: "Swipe ← → to watch all three reels. The odd one is easier to spot once you've seen the other two.",
          cta: "Got it",
        }
      : {
          kicker: "no rush",
          title: "Feeling it?",
          body: "Snap guesses don't help the map. If a set doesn't click, it's totally fine to stop and come back later — quality beats speed here.",
          cta: "Keep going",
        };
  return (
    <div
      onPointerDown={(e) => e.stopPropagation()}
      className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-5 bg-bg-canvas/92 px-9 text-center backdrop-blur"
    >
      <p className="font-mono text-[10px] uppercase tracking-[0.3em] text-accent">
        {copy.kicker}
      </p>
      <h2 className="font-display text-[26px] font-extrabold leading-tight text-fg-default">
        {copy.title}
      </h2>
      <p className="max-w-xs font-mono text-[12px] leading-relaxed text-fg-muted">
        {copy.body}
      </p>
      <button
        type="button"
        onClick={onDismiss}
        className="mt-2 h-12 rounded-bar bg-accent px-7 font-display text-[15px] font-semibold uppercase tracking-wide text-bg-canvas active:translate-y-0.5"
      >
        {copy.cta}
      </button>
    </div>
  );
}

/**
 * Shown when someone is labouring over judgments (slowing down / very long
 * answers). Reassures them it's fine to stop, and — inside Telegram — offers a
 * "remind me later" button that schedules a come-back DM ~20h out.
 */
function RestNudge({ onDismiss }: { onDismiss: () => void }) {
  const [pending, setPending] = useState(false);
  const [reminded, setReminded] = useState(false);
  const canRemind = isTelegram();

  const onRemind = async () => {
    setPending(true);
    const ok = await remindLater(config.apiBaseUrl);
    setPending(false);
    if (ok) setReminded(true);
    else onDismiss(); // couldn't schedule — don't trap them, just let them go on
  };

  return (
    <div
      onPointerDown={(e) => e.stopPropagation()}
      className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-5 bg-bg-canvas/92 px-9 text-center backdrop-blur"
    >
      <p className="font-mono text-[10px] uppercase tracking-[0.3em] text-accent">
        take a breath
      </p>
      <h2 className="font-display text-[26px] font-extrabold leading-tight text-fg-default">
        Been at it a while?
      </h2>
      <p className="max-w-xs font-mono text-[12px] leading-relaxed text-fg-muted">
        No rush at all — your eye gets tired and that&apos;s totally fine. Stop
        whenever you like; the sets will keep for next time.
      </p>
      {reminded ? (
        <>
          <p className="font-mono text-[12px] leading-relaxed text-affirm">
            ok — i&apos;ll nudge you to come back later 💛
          </p>
          <button
            type="button"
            onClick={onDismiss}
            className="mt-1 h-12 rounded-bar bg-accent px-7 font-display text-[15px] font-semibold uppercase tracking-wide text-bg-canvas active:translate-y-0.5"
          >
            Done
          </button>
        </>
      ) : (
        <div className="mt-2 flex w-full max-w-xs flex-col items-center gap-3">
          {canRemind && (
            <button
              type="button"
              onClick={() => void onRemind()}
              disabled={pending}
              className="h-12 w-full rounded-bar bg-accent px-7 font-display text-[15px] font-semibold uppercase tracking-wide text-bg-canvas active:translate-y-0.5 disabled:opacity-60"
            >
              {pending ? "Setting…" : "Remind me later"}
            </button>
          )}
          <button
            type="button"
            onClick={onDismiss}
            className="font-mono text-[11px] uppercase tracking-[0.16em] text-fg-muted"
          >
            {canRemind ? "keep going" : "ok, keep going"}
          </button>
        </div>
      )}
    </div>
  );
}

function StatusScreen({ status, onRetry }: { status: string; onRetry: () => void }) {
  const copy =
    status === "loading"
      ? "Loading comparisons…"
      : status === "error"
        ? "Couldn't reach the server."
        : status === "exhausted"
          ? "All caught up — no more comparisons."
          : "Getting ready…";
  return (
    <div className="flex h-full flex-col items-center justify-center gap-5 px-8 text-center">
      <div
        className="h-10 w-10 rounded-full border-2 border-white/15 border-t-accent"
        style={status === "loading" ? { animation: "spin 0.9s linear infinite" } : undefined}
      />
      <p className="font-display text-lg text-fg-default">{copy}</p>
      {status === "error" && (
        <ActionButton variant="continue" onClick={onRetry}>
          Retry
        </ActionButton>
      )}
    </div>
  );
}
