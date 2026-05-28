import {
  useId,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { tokens } from "@/ui/tokens";
import { SectionHeading } from "./SectionHeading";

type Props = {
  index: string;
  title: string;
  children: ReactNode;
  /** Defaults to open. Persists user toggles for the lifetime of the
   * section's React mount; re-mounting the pane resets to default. */
  defaultOpen?: boolean;
};

/**
 * Inspector section frame with a user-toggleable open/close animation.
 *
 * The pane-level open/close/swap choreography is handled upstream by
 * `Inspector` (a single content wrapper slides + fades). Sections
 * themselves are static at the pane level — they always render in
 * their final state when the pane appears, and the user can collapse
 * or expand any individual section by clicking its chevron.
 *
 * Animation: `grid-template-rows: 0fr ↔ 1fr` + `opacity`, with no
 * stagger. Single source of timing in `tokens.inspector.motion`.
 */
export const CollapsibleSection = ({
  index,
  title,
  children,
  defaultOpen = true,
}: Props) => {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();
  const motion = tokens.inspector.motion;
  const ease = tokens.motion.easeOut;
  const trans = `${motion.contentMs}ms ${ease}`;

  return (
    <section>
      <div style={headerRow}>
        <div style={headerFlex}>
          <SectionHeading index={index}>{title}</SectionHeading>
        </div>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls={bodyId}
          style={toggleBtn}
        >
          <span aria-hidden="true" style={chevron(open, motion.contentMs, ease)}>›</span>
        </button>
      </div>
      <div
        id={bodyId}
        aria-hidden={!open}
        style={{
          display: "grid",
          gridTemplateRows: open ? "1fr" : "0fr",
          transition: `grid-template-rows ${trans}, opacity ${trans}`,
          opacity: open ? 1 : 0,
          pointerEvents: open ? "auto" : "none",
        }}
      >
        <div style={{ overflow: "hidden" }}>{children}</div>
      </div>
    </section>
  );
};

const headerRow: CSSProperties = {
  display: "flex",
  alignItems: "baseline",
  justifyContent: "space-between",
  gap: 12,
};

const headerFlex: CSSProperties = {
  flex: 1,
  minWidth: 0,
};

const toggleBtn: CSSProperties = {
  flex: "0 0 auto",
  alignSelf: "center",
  cursor: "pointer",
  background: "transparent",
  border: "none",
  padding: 4,
  margin: 0,
  color: tokens.ink.faint,
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  outline: "none",
  WebkitTapHighlightColor: "transparent",
};

const chevron = (open: boolean, dur: number, ease: string): CSSProperties => ({
  display: "inline-block",
  fontFamily: tokens.type.mono,
  fontSize: 14,
  lineHeight: 1,
  transform: open ? "rotate(90deg)" : "rotate(0deg)",
  transition: `transform ${dur}ms ${ease}`,
});
