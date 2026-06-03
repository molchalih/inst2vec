import { useTrackingMode, useToggleTracking, useIsIntroPlaying, useIsTransitioning } from "@/state";

/**
 * Point-tracking toggle glyph. Toggles tracking mode (the toggle action clears
 * the tracked creator when turning off). An armed/off look only — no third
 * "tracking a point" treatment. Disabled-faded (not hidden) during the intro
 * and a version-switch transition, mirroring the pill row's guard.
 *
 * Position-agnostic: this is just the glyph button. Its placement is owned by
 * the dock that hosts it; the control never reacts to inspector/panel state.
 */
export const TrackingControl = () => {
  const mode = useTrackingMode();
  const toggle = useToggleTracking();
  const introPlaying = useIsIntroPlaying();
  const transitioning = useIsTransitioning();
  const disabled = introPlaying || transitioning;

  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={mode}
      aria-label="Toggle point tracking"
      disabled={disabled}
      className={[
        "pointer-events-auto",
        "w-dock-control h-dock-control",
        "grid place-items-center",
        "transition duration-medium ease-motion-out",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-fg-default/60",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        mode
          ? "text-fg-default"
          : "text-fg-muted opacity-70 hover:text-fg-default hover:opacity-100",
      ].join(" ")}
    >
      <svg
        className="w-dock-icon h-dock-icon"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={1.75}
        strokeLinecap="round"
        aria-hidden="true"
      >
        <circle cx="12" cy="12" r="6.5" />
        <circle cx="12" cy="12" r="1.5" fill="currentColor" stroke="none" />
        <line x1="12" y1="1.5" x2="12" y2="5" />
        <line x1="12" y1="19" x2="12" y2="22.5" />
        <line x1="1.5" y1="12" x2="5" y2="12" />
        <line x1="19" y1="12" x2="22.5" y2="12" />
      </svg>
    </button>
  );
};
