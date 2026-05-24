import { tokens } from "@/ui/tokens";

type SkeletonProps = { width?: string | number; height?: string | number };

/**
 * Subtle pulsing placeholder used while detail data is loading. Same
 * physical footprint as the real content so the layout doesn't shift
 * on arrival.
 */
export const Skeleton = ({ width = "100%", height = tokens.inspector.bar.height }: SkeletonProps) => (
  <div
    aria-hidden="true"
    style={{
      width, height,
      borderRadius: tokens.inspector.bar.radius,
      background: `rgb(255 255 255 / ${tokens.inspector.bar.trackAlpha})`,
      animation: `inspector-skeleton-pulse ${tokens.inspector.skeleton.pulseMs}ms ease-in-out infinite`,
    }}
  />
);
