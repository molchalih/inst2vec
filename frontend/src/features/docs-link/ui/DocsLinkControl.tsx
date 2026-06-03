// The rendered paper, hosted on the CDN. A static external page, so this is a
// plain constant rather than build-time config — there is one and only one
// target.
const PAPER_URL = "https://cdn.240.agency/inst2vec/";

/**
 * Paper glyph that opens the rendered study in a new tab. A borderless
 * floating link sharing the dock's glyph sizing and the tracking control's
 * resting/hover treatment. No pressed or disabled state — it is a navigation
 * affordance, not a toggle; the dock owns its entrance and placement.
 */
export const DocsLinkControl = () => (
  <a
    href={PAPER_URL}
    target="_blank"
    rel="noopener noreferrer"
    aria-label="Read the paper"
    className={[
      "pointer-events-auto",
      "w-dock-control h-dock-control",
      "grid place-items-center",
      "transition duration-medium ease-motion-out",
      "focus-visible:outline focus-visible:outline-2 focus-visible:outline-fg-default/60",
      "text-fg-muted opacity-70 hover:text-fg-default hover:opacity-100",
    ].join(" ")}
  >
    <svg
      className="w-dock-icon h-dock-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="15.5" y1="13" x2="8.5" y2="13" />
      <line x1="15.5" y1="16.5" x2="8.5" y2="16.5" />
      <line x1="10.5" y1="9.5" x2="8.5" y2="9.5" />
    </svg>
  </a>
);
