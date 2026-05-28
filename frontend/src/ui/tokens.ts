/**
 * Single source of truth for visual constants. Consumed by:
 *   1. tailwind.config.ts via CSS variables (see styles.css)
 *   2. ui/primitives/* and features/* /ui/* via Tailwind classes
 *   3. render/draw/* by direct import (Pixi cannot see CSS)
 */
export const tokens = {
  bg: {
    canvas: "#0b1220",
  },
  fg: {
    default: "#e2e8f0",
    muted: "#94a3b8",
  },
  // Inspector type system. Two faces, deliberately: an editorial
  // display serif for names and a precise monospace for every label
  // and numeral (the "field report" voice). The rest of the chrome
  // stays on the system sans stack — these are scoped to the panel.
  type: {
    serif: '"Fraunces", "Iowan Old Style", Georgia, serif',
    mono: '"Spline Sans Mono", ui-monospace, "SFMono-Regular", Menlo, monospace',
  },
  // Slate ink ramp used across the inspector. `fg.*` stays the
  // Tailwind-bridged pair for the rest of the app; this is the fuller
  // scale the panel's typographic hierarchy needs. `line` is the
  // hairline-rule colour.
  ink: {
    bright: "#f1f5f9",
    default: "#e2e8f0",
    dim: "#cbd5e1",
    muted: "#94a3b8",
    faint: "#64748b",
    line: "rgb(255 255 255 / 0.07)",
  },
  noise: {
    alpha: 0.5,
  },
  dot: {
    radius: 4,
    radiusHover: 7,
    alpha: 0.7,
    strokeColorHover: "#ffffff",
    strokeWidthHover: 2,
    // Aesthetic-centrality size encoding. HDBSCAN soft membership [0, 1]
    // maps through a convex gamma curve to a per-user radius multiplier
    // in [min, max]. The real distribution is heavily skewed near 1
    // (median ≈ 0.95), so a linear map collapses every signal user to
    // near-max; gamma > 1 spreads them out. Noise points bypass this
    // mapping and stay at scale 1.
    centrality: {
      min: 0.6,
      max: 1.5,
      gamma: 3,
    },
  },
  ellipse: {
    strokeWidth: 2,
    strokeWidthHover: 2.25,
    strokeAlpha: 0.3,
    strokeAlphaHover: 0.7,
    fillAlpha: 0.05,
  },
  hover: {
    dotRadiusPx: 12, // tolerance to hover, where it triggers
  },
  motion: {
    fast: 120,
    medium: 220,
    slow: 420,
    easeOut: "cubic-bezier(0.22, 1, 0.36, 1)",
    versionSwitch: {
      phase1: 900,  // camera eases from user's pan/zoom to fit-of-from-run
      phase2: 400,  // cluster ellipses fade out (from-side dissolves)
      // Phase 3 is the dot-morph window. It splits into a uniform
      // eased flight (positions move) followed by a per-dot radial
      // wave-pulse (color + scale-pop, fans outward from the new
      // arrangement's world origin). Each sub-window is tuned in ms
      // independently — change one without affecting the other.
      phase3: {
        flightMs: 2300,  // uniform position morph (easeOutCubic)
        pulseMs: 1650,   // per-dot wave-pulse (color + scalePop)
      },
      phase4: 750,  // cluster ellipses fade in (to-side appears)
    },
    // Per-dot radial wave that fans out from the world origin during
    // the pulse sub-window of phase 2. `spread` is each dot's own
    // window as a fraction of pulseMs; smaller = sharper wavefront.
    // `jitter` adds wobble (deterministic per dot id, see
    // core/morph/wave.ts). The scale-pop pair matches atlas/app:
    // ~0.1s grow, ~0.3s settle, so upFrac = 0.1 / (0.1 + 0.3).
    wave: {
      spread: 0.2,
      jitter: 0.3,
      scalePopPeak: 1.7,
      scalePopUpFrac: 0.4,
    },
    cameraFocus: {
      durationMs: 1200,
    },
    // One-time page-load entrance. Dots fade in stacked at the
    // screen-center world point (fadeMs), fan out to their fitted
    // positions with a per-dot random launch delay of up to
    // maxStaggerFrac of the flight window (flightMs), then cluster
    // ellipses fade in (settleMs). Fires once per load; never on a
    // version switch. See interaction/useIntroAnimation.ts.
    intro: {
      fadeMs: 1250,
      flightMs: 2000,
      settleMs: 750,
      maxStaggerFrac: 0.4,
    },
  },
  drawer: {
    height: 56,
    paddingX: 16,
    gap: 8,
    // Breathing room above the tongue's glyph. When the drawer is
    // closed, the tongue sits at y=0; without this offset the "+"
    // touches the top of the viewport.
    tonguePt: 8,
  },
  pill: {
    paddingX: 14,
    paddingY: 6,
    radius: 999,
  },
  glass: {
    blurPx: 12,
    shadow: "0 8px 24px rgb(0 0 0 / 0.25)",
  },
  tooltip: {
    paddingX: 10,
    paddingY: 6,
    radius: 8,
  },
  palette: {
    cluster: [
      "#d2d8d9", "#969293", "#e96f51", "#80c470",
      "#ffeb62", "#4196ff", "#fd568a", "#53cdbd",
      "#bc7ad5", "#ee9931", "#fe7ed3", "#ceea90",
      "#2cd8f6", "#4abbf9", "#fabf32", "#14a9b1",
    ],
    noise: "#576175",
  },
  viewport: {
    clamp: {
      minScaleFactor: 1,
      maxScaleFactor: 5,
      panMarginPx: 750,
    },
  },
  panel: {
    widthPx: 372,
    // Deepened from slate-700/.82 to slate-900/.86: the serif display
    // type and hairline rules need more contrast under them than the
    // old lighter glass gave.
    bg: "rgb(13 20 36 / 0.86)",
    paddingX: 18,
    paddingY: 16,
    sectionGap: 18,
    close: { top: 12, right: 14, size: 18 },
  },
  inspector: {
    // Inspector open/close + pane-swap choreography. Decoupled from
    // the global `motion` ramp so the panel can move at its own pace.
    // Consumed by `useInspectorChoreography`, `Inspector`, and
    // `Panel` (via the `durationMs` prop on the inspector mount).
    motion: {
      slideMs: 560,      // panel slide-in / slide-out
      contentMs: 560,    // pane content slide-in / slide-out (single block)
      // Accent morph runs slightly past one full swap so the gradient
      // continues to settle as the new pane finishes sliding in.
      accentMs: 1150,
      /**
       * Fraction of the first phase's duration after which the second
       * phase begins. 0 = strictly sequential (no overlap), 1 =
       * second phase starts immediately. Applied only to the
       * open and close paths (where two distinct DOM elements — the
       * panel and the content wrapper — can animate concurrently).
       * Swap paths stay sequential because both halves share the same
       * content element.
       */
      overlap: 0.35,
    },
    bar:       { height: 6, radius: 999, trackAlpha: 0.07 },
    chip: {
      paddingX: 10, paddingY: 4, radius: 999,
      weightFillAlpha: 0.22,
    },
    // Colour palettes for clip-label tag chips. One hue family with
    // three weight steps so the three kinds read as a hierarchy.
    // Consumed by features/inspector/ui/primitives/Chip via its
    // `tone` prop. `warningOutline` overrides the border colour on
    // any chip belonging to a warn-validation clip.
    tagChip: {
      observable: {
        bg: "var(--bg-chip-observable)",
        fg: "var(--fg-chip-observable)",
        border: "var(--border-chip-observable)",
      },
      aesthetic: {
        bg: "var(--bg-chip-aesthetic)",
        fg: "var(--fg-chip-aesthetic)",
        border: "var(--border-chip-aesthetic)",
      },
      community: {
        bg: "var(--bg-chip-community)",
        fg: "var(--fg-chip-community)",
        border: "var(--border-chip-community)",
      },
      warningOutline: "var(--border-chip-warn)",
    },
    // Layout knobs for SectionClips row cards.
    clipCard: {
      padding: 8,
      gap: 8,
      thumbWidth: 80,
      radius: 6,
    },
    microBar: {
      labelColWidth: 80, valueColWidth: 28, rowGap: 6,
      moodColor:   "#bc7ad5",
      timbreColor: "#80c470",
    },
    neighbor: {
      paddingX: 9, paddingY: 7, radius: 7,
      bgAlpha: 0.03, bgAlphaHover: 0.06,
    },
    skeleton: { pulseMs: 1200 },
    crossfadeMs: 150,
    langChip: {
      paddingX: 7, paddingY: 3, radius: 4,
      bgAlpha: 0.04,
    },
    // Full-bleed top→bottom accent wash painted across the whole panel
    // (behind the content, not on an inner padded box). `topPct` is the
    // accent strength at the top; it fades to transparent by `fadeStop`
    // down the height. Reads the live `--accent` CSS var.
    wash: { topPct: 16, fadeStop: 80 },
    // Header block: serif name (forced to a display optical size for
    // the high-contrast cut), mono meta, and a closing hairline.
    header: { nameSize: 23, nameWeight: 500, nameOpsz: 64,
              metaSize: 11, ruleGap: 15 },
    // Indexed section heads: a mono "01" in the accent colour, an
    // em-dash, then the tracked-out uppercase label. Sections past the
    // first carry a top hairline.
    sectionHead: { size: 10, tracking: "0.16em", indexGap: 8,
                   gapTop: 18, gapBottom: 11 },
  },
  interaction: {
    focus: {
      runFitPadding: 0.1,
      clusterFitPadding: 0.1,
      creatorScaleFactor: 4,
    },
  },
} as const;

export type Tokens = typeof tokens;
