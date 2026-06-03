import type { Config } from "tailwindcss";
import { tokens } from "./src/ui/tokens";

const px = (n: number) => `${n}px`;
const ms = (n: number) => `${n}ms`;

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: { canvas: "rgb(var(--bg-canvas) / <alpha-value>)" },
        fg: {
          default: "rgb(var(--fg-default) / <alpha-value>)",
          muted: "rgb(var(--fg-muted) / <alpha-value>)",
        },
      },
      spacing: {
        "drawer-h": px(tokens.drawer.height),
        "drawer-px": px(tokens.drawer.paddingX),
        "drawer-gap": px(tokens.drawer.gap),
        "tongue-pt": px(tokens.drawer.tonguePt),
        "pill-px": px(tokens.pill.paddingX),
        "pill-py": px(tokens.pill.paddingY),
        "tooltip-px": px(tokens.tooltip.paddingX),
        "tooltip-py": px(tokens.tooltip.paddingY),
        "dock-offset": px(tokens.dock.offset),
        "dock-gap": px(tokens.dock.gap),
        "dock-control": px(tokens.dock.control.size),
        "dock-icon": px(tokens.dock.control.iconSize),
        "dock-reveal-x": px(tokens.dock.revealX),
      },
      borderRadius: {
        pill: `${tokens.pill.radius}px`,
        tooltip: `${tokens.tooltip.radius}px`,
      },
      backdropBlur: {
        glass: `${tokens.glass.blurPx}px`,
      },
      boxShadow: {
        glass: tokens.glass.shadow,
      },
      transitionDuration: {
        fast: ms(tokens.motion.fast),
        medium: ms(tokens.motion.medium),
        slow: ms(tokens.motion.slow),
        chrome: ms(tokens.motion.chrome.slideMs),
      },
      transitionTimingFunction: {
        "motion-out": tokens.motion.easeOut,
        chrome: tokens.motion.easeChrome,
      },
    },
  },
  plugins: [],
};
export default config;
