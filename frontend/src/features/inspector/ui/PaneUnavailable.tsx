import { tokens } from "@/ui/tokens";

/**
 * Quiet fallback shown when the active run ships no detail payloads
 * (manifest details_available=false). The basic header still renders
 * above this; the detail sections are omitted entirely so there is no
 * perpetual loading skeleton for data that will never arrive.
 */
export const PaneUnavailable = () => (
  <div style={{
    marginTop: tokens.inspector.sectionHead.gapTop,
    fontSize: 12, color: tokens.ink.muted,
    fontFamily: tokens.type.mono, lineHeight: 1.5,
  }}>
    Detailed metrics aren&rsquo;t available for this map.
  </div>
);
