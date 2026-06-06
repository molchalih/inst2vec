import type { ReactNode } from "react";

type Variant = "continue" | "skip";

interface ActionButtonProps {
  children: ReactNode;
  variant: Variant;
  disabled?: boolean;
  onClick: () => void;
}

const VARIANTS: Record<Variant, string> = {
  continue:
    "bg-affirm text-bg-canvas shadow-card active:translate-y-0.5 disabled:bg-bg-raised disabled:text-fg-faint disabled:shadow-none",
  skip: "bg-transparent text-fg-muted border border-white/12 active:translate-y-0.5",
};

/** A chunky, tactile action-bar button. */
export function ActionButton({
  children,
  variant,
  disabled = false,
  onClick,
}: ActionButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={[
        "h-14 rounded-bar px-6 font-display text-[15px] font-semibold uppercase tracking-wide",
        "transition duration-fast ease-out select-none disabled:cursor-not-allowed",
        VARIANTS[variant],
      ].join(" ")}
    >
      {children}
    </button>
  );
}
