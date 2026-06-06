/**
 * Shown when no access code is present in the URL or storage (auth gate). The app
 * opens nothing without a valid invite link.
 */
export function AccessRequired() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-5 px-9 text-center">
      <p className="font-mono text-[10px] uppercase tracking-[0.3em] text-accent">
        triage
      </p>
      <h1 className="font-display text-[28px] font-extrabold leading-tight text-fg-default">
        Invite only
      </h1>
      <p className="max-w-xs font-mono text-[12px] leading-relaxed text-fg-muted">
        This tool opens from a personal invite. Open it from the Telegram bot
        using the start link a friend sent you, or use a personal web link
        (<span className="text-fg-default">…/?code=YOURCODE</span>) —
        either way it&apos;ll remember you on this device.
      </p>
    </div>
  );
}
