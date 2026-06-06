import { useState } from "react";
import type { CreatorCard as CreatorCardData } from "@/data";
import { usePauseWhenInactive } from "@/interaction";
import { CardShell, Chip, groupHue } from "@/ui";
import { captionKeywords, audioTerms } from "../digest";

interface CreatorCardProps {
  card: CreatorCardData;
  /** position label within the comparison: "A" | "B" | "C" */
  label: string;
  isOdd: boolean;
  /** true when this is the front card — only it decodes/plays video */
  active: boolean;
  /** whether the browser will allow reel audio yet (after a user gesture) */
  audioUnlocked: boolean;
  /** toggle this creator as the odd one (tap on touch; this is the a11y path) */
  onToggleOdd: () => void;
  onSkip: () => void;
}

const stop = (e: { stopPropagation: () => void }) => e.stopPropagation();

export function CreatorCard({
  card,
  label,
  isOdd,
  active,
  audioUnlocked,
  onToggleOdd,
  onSkip,
}: CreatorCardProps) {
  const hue = groupHue(card.seed_group);
  const captions = captionKeywords(card);
  const audio = audioTerms(card);
  const clip = card.clips[0];
  const group = card.seed_group ?? "unsorted";
  const soundOn = active && audioUnlocked;
  const videoRef = usePauseWhenInactive(active, !soundOn);
  const [infoOpen, setInfoOpen] = useState(false);

  return (
    <CardShell crossed={isOdd}>
      <div className="relative h-full w-full overflow-hidden">
        {/* Accessible equivalent of tapping the card to mark the odd one — kept
            visually hidden so the touch UX (tap the card) is unchanged. */}
        <button
          type="button"
          onPointerDown={stop}
          onClick={onToggleOdd}
          aria-pressed={isOdd}
          className="sr-only"
        >
          mark {group} as the odd one
        </button>
        {/* Layered so there is never a black flash on swipe: a tinted base, the
            poster image, then the video on top. The video keeps its last frame
            when inactive (see usePauseWhenInactive) and the poster sits behind it,
            so the card always shows imagery while the next frame decodes. */}
        <div
          className="absolute inset-0"
          style={{
            background: `radial-gradient(120% 90% at 70% 10%, ${hue}33 0%, rgb(11 11 15) 62%)`,
          }}
        />
        {clip?.poster_url && (
          <div
            className="absolute inset-0 bg-cover bg-center"
            style={{ backgroundImage: `url(${clip.poster_url})` }}
          />
        )}
        {clip?.video_url && (
          <video
            ref={videoRef}
            className="pointer-events-none absolute inset-0 h-full w-full object-cover"
            src={clip.video_url}
            poster={clip.poster_url ?? undefined}
            loop
            playsInline
            preload={active ? "auto" : "metadata"}
          />
        )}

        {/* Overlays live in their own GPU-composited layer (translateZ + z-10 +
            isolation) so a starting inline video on iOS — which gets promoted to
            a hardware layer — can never paint over the title/controls. */}
        <div
          className="absolute inset-0 z-10"
          style={{ transform: "translateZ(0)", isolation: "isolate" }}
        >
        {/* legibility scrims */}
        <div className="pointer-events-none absolute inset-x-0 top-0 h-28 bg-gradient-to-b from-bg-canvas/70 to-transparent" />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-1/2 bg-gradient-to-t from-bg-canvas via-bg-canvas/45 to-transparent" />

        <span className="absolute left-4 top-4 rounded bg-bg-canvas/65 px-1.5 py-0.5 font-mono text-[11px] uppercase tracking-[0.3em] text-fg-default/90">
          {label}
        </span>

        <button
          type="button"
          onPointerDown={stop}
          onClick={onSkip}
          className="absolute right-3 top-3 rounded-chip border border-white/20 bg-bg-canvas/70 px-3 py-1.5 font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted transition duration-fast ease-out active:scale-95"
        >
          skip
        </button>

        <div className="absolute bottom-4 left-4 right-16">
          <div className="inline-block rounded-lg bg-bg-canvas/80 px-2.5 py-1.5">
            <div className="font-display text-[22px] font-extrabold leading-tight text-fg-default">
              {group}
            </div>
            <div className="mt-0.5 font-mono text-[10px] uppercase tracking-[0.2em] text-fg-muted">
              #{card.creator_id}
              {clip?.video_url && !soundOn ? " · tap to start" : ""}
            </div>
          </div>
        </div>

        <button
          type="button"
          onPointerDown={stop}
          onClick={() => setInfoOpen((o) => !o)}
          aria-label="creator details"
          className="absolute bottom-4 right-4 flex h-10 w-10 items-center justify-center rounded-full border border-white/25 bg-bg-canvas/70 font-display text-[15px] italic text-fg-default active:scale-95"
        >
          i
        </button>

        {/* slide-up detail sheet */}
        <div
          onPointerDown={stop}
          className={[
            "absolute inset-x-0 bottom-0 max-h-[68%] touch-pan-y space-y-5 overflow-y-auto rounded-t-card border-t border-white/10",
            "bg-bg-surface p-card-p transition-transform duration-medium ease-out",
            infoOpen ? "translate-y-0" : "translate-y-full",
          ].join(" ")}
        >
          <div className="flex items-center justify-between">
            <h3 className="font-display text-lg font-bold text-fg-default">{group}</h3>
            <button
              type="button"
              onClick={() => setInfoOpen(false)}
              className="font-mono text-[10px] uppercase tracking-[0.14em] text-fg-muted"
            >
              close
            </button>
          </div>

          <Section index="01" title="caption signal">
            {captions.length ? (
              <div className="flex flex-wrap gap-2">
                {captions.map((k) => (
                  <Chip key={k}>{k}</Chip>
                ))}
              </div>
            ) : (
              <Empty>no captions in digest yet</Empty>
            )}
          </Section>

          <Section index="02" title="audio signal">
            {audio.length ? (
              <div className="flex flex-wrap gap-2">
                {audio.map((t) => (
                  <Chip key={t} hue={hue}>
                    {t}
                  </Chip>
                ))}
              </div>
            ) : (
              <Empty>audio digest pending</Empty>
            )}
          </Section>
        </div>
        </div>
      </div>
    </CardShell>
  );
}

function Section({
  index,
  title,
  children,
}: {
  index: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <h4 className="mb-2.5 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-fg-faint">
        <span className="text-accent">{index}</span>
        <span>—</span>
        <span>{title}</span>
      </h4>
      {children}
    </section>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="font-mono text-[11px] lowercase text-fg-faint">{children}</p>;
}
