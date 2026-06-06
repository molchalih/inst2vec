import type { Config } from "tailwindcss";
import { tokens } from "./src/ui/tokens";

const px = (n: number) => `${n}px`;
const ms = (n: number) => `${n}ms`;

// Colours bridged via CSS variables (channel form, see index.html :root) get
// slash-opacity support; one-off accents are plain hex from tokens.
const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          canvas: "rgb(var(--bg-canvas) / <alpha-value>)",
          surface: "rgb(var(--bg-surface) / <alpha-value>)",
          raised: "rgb(var(--bg-raised) / <alpha-value>)",
        },
        fg: {
          default: "rgb(var(--fg-default) / <alpha-value>)",
          muted: "rgb(var(--fg-muted) / <alpha-value>)",
          faint: "rgb(var(--fg-faint) / <alpha-value>)",
        },
        accent: {
          DEFAULT: "rgb(var(--accent) / <alpha-value>)",
          bright: tokens.accent.bright,
          deep: tokens.accent.deep,
        },
        affirm: {
          DEFAULT: "rgb(var(--affirm) / <alpha-value>)",
          deep: tokens.affirm.deep,
        },
      },
      fontFamily: {
        display: tokens.type.display.split(",").map((s) => s.trim().replace(/^"|"$/g, "")),
        mono: tokens.type.mono.split(",").map((s) => s.trim().replace(/^"|"$/g, "")),
      },
      spacing: {
        "card-p": px(tokens.card.padding),
        "card-gap": px(tokens.card.gap),
        "bar-h": px(tokens.actionBar.height),
        "bar-gap": px(tokens.actionBar.gap),
        "chip-px": px(tokens.chip.paddingX),
        "chip-py": px(tokens.chip.paddingY),
      },
      borderRadius: {
        card: px(tokens.card.radius),
        bar: px(tokens.actionBar.buttonRadius),
        chip: px(tokens.chip.radius),
      },
      boxShadow: {
        card: tokens.card.shadow,
      },
      transitionDuration: {
        fast: ms(tokens.motion.fast),
        medium: ms(tokens.motion.medium),
        slow: ms(tokens.motion.slow),
      },
      transitionTimingFunction: {
        out: tokens.motion.easeOut,
        pop: tokens.motion.easePop,
      },
    },
  },
  plugins: [],
};
export default config;
