import { describe, expect, it, vi } from "vitest";
import type { Graphics } from "pixi.js";
import { tokens } from "@/ui/tokens";
import { drawEllipses } from "./drawEllipses";
import type { DrawableEllipse, EllipsesFrame } from "../frame";

const mockGraphics = () => {
  const polys: number[] = [];
  const strokes: { color: string; alpha: number; width: number }[] = [];
  const fills: { color: string; alpha: number }[] = [];
  const clears = { count: 0 };
  const g = {
    clear: vi.fn(() => { clears.count += 1; return g; }),
    poly: vi.fn(() => { polys.push(1); return g; }),
    fill: vi.fn((o: { color: string; alpha: number }) => { fills.push(o); return g; }),
    stroke: vi.fn((o: { color: string; alpha: number; width: number }) => { strokes.push(o); return g; }),
  };
  return { g: g as unknown as Graphics, polys, strokes, fills, clears };
};

const e = (over: Partial<DrawableEllipse> = {}): DrawableEllipse => ({
  id: 0, cx: 0, cy: 0, rx: 1, ry: 0.5, angle: 0, color: "#ffffff", ...over,
});

const frame = (
  ellipses: DrawableEllipse[],
  over: Partial<EllipsesFrame> = {},
): EllipsesFrame => ({
  ellipses, alphaScale: 1, strokeWidthScale: 1, ...over,
});

describe("drawEllipses", () => {
  it("clears the graphics on every call", () => {
    const m = mockGraphics();
    drawEllipses(m.g, frame([]), { x: 0, y: 0, scale: 1 });
    expect(m.clears.count).toBe(1);
    expect(m.polys).toHaveLength(0);
  });

  it("draws one polygon per ellipse in the frame", () => {
    const m = mockGraphics();
    drawEllipses(m.g, frame([e({ id: 0 }), e({ id: 1 })]), { x: 0, y: 0, scale: 1 });
    expect(m.polys).toHaveLength(2);
  });

  it("inverse-scales the stroke width by viewport.scale and frame.strokeWidthScale", () => {
    const m = mockGraphics();
    drawEllipses(m.g, frame([e()], { strokeWidthScale: 2 }), { x: 0, y: 0, scale: 4 });
    // tokens.ellipse.strokeWidth (2) * strokeWidthScale (2) / scale (4) = 1
    expect(m.strokes[0]!.width).toBeCloseTo(1, 6);
  });

  it("multiplies stroke + fill alpha by frame.alphaScale", () => {
    const m = mockGraphics();
    drawEllipses(m.g, frame([e()], { alphaScale: 0.5 }), { x: 0, y: 0, scale: 1 });
    expect(m.fills[0]!.alpha).toBeCloseTo(tokens.ellipse.fillAlpha * 0.5, 6);
    expect(m.strokes[0]!.alpha).toBeCloseTo(tokens.ellipse.strokeAlpha * 0.5, 6);
  });

  it("uses the resolved color from each drawable", () => {
    const m = mockGraphics();
    drawEllipses(m.g, frame([e({ color: "#abcdef" })]), { x: 0, y: 0, scale: 1 });
    expect(m.fills[0]!.color).toBe("#abcdef");
    expect(m.strokes[0]!.color).toBe("#abcdef");
  });
});
