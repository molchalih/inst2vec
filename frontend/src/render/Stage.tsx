import { Application, extend } from "@pixi/react";
import { Container, Graphics } from "pixi.js";
import { useAtomValue } from "jotai";
import { useRef, type ReactNode } from "react";
import { usePanZoom, useHover, useClick } from "@/interaction";
import { hoverAtom, stretchedRunAtom, viewportAtom } from "@/state";
import { tokens } from "@/ui/tokens";

extend({ Container, Graphics });

type StageProps = { children: ReactNode };

/**
 * Pixi <Application> + a single transformed container holding the
 * viewport transform. Pointer input goes through the wrapper <div>,
 * not Pixi.
 *
 * The <Application> mount is gated on stretchedRunAtom being non-null.
 * @pixi/react's Application captures its children JSX on first mount
 * and awaits an asynchronous Pixi.init(); when init resolves it
 * re-applies the captured JSX, which would clobber any newer commits
 * that happened during init. Gating on run-availability means the
 * captured first-mount JSX already has the fitted viewport (derived
 * from run + size in viewportAtom) and the loaded run's content, so
 * the post-init re-apply is a no-op.
 */
export const Stage = ({ children }: StageProps) => {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  usePanZoom(wrapperRef);
  useHover(wrapperRef);
  useClick(wrapperRef);
  const run = useAtomValue(stretchedRunAtom);
  const viewport = useAtomValue(viewportAtom);
  const hover = useAtomValue(hoverAtom);
  const cursorClass = hover.dotId === null ? "" : "cursor-pointer";

  if (run) {
    const isBrowser = typeof globalThis !== "undefined";
    const resolution = isBrowser ? globalThis.devicePixelRatio : 1;

    return (
      <div ref={wrapperRef} className={`absolute inset-0 ${cursorClass}`}>
        <Application
          background={tokens.bg.canvas}
          antialias
          resolution={resolution}
          autoDensity
          {...(isBrowser ? { resizeTo: globalThis as unknown as Window } : {})}
        >
          <pixiContainer
            x={viewport.x}
            y={viewport.y}
            scale={viewport.scale}
            sortableChildren
          >
            {children}
          </pixiContainer>
        </Application>
      </div>
    );
  }

  return <div ref={wrapperRef} className="absolute inset-0" />;
};
