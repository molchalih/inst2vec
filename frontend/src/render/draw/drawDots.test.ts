import { describe, expect, it, vi } from "vitest";
import type { AlphaFilter, Filter, Graphics } from "pixi.js";
import { drawDots } from "./drawDots";
import type { DotsFrame, DrawableUser } from "../frame";

type CircleCall = { x: number; y: number; r: number };
type FillCall = { color: string; alpha: number };

const mockGraphics = () => {
  const circles: CircleCall[] = [];
  const fills: FillCall[] = [];
  const clears = { count: 0 };
  const g = {
    alpha: 1,
    filters: null as Filter[] | null,
    clear: vi.fn(() => { clears.count += 1; return g; }),
    circle: vi.fn((x: number, y: number, r: number) => {
      circles.push({ x, y, r }); return g;
    }),
    fill: vi.fn((opts: FillCall) => { fills.push(opts); return g; }),
  };
  return { g: g as unknown as Graphics, raw: g, circles, fills, clears };
};

// Stand-in for pixi's AlphaFilter: drawDots only reads/writes `.alpha`.
const mockFade = () => ({ alpha: 1 } as unknown as AlphaFilter);

const u = (over: Partial<DrawableUser>): DrawableUser => ({
  id: 0, x: 0, y: 0, color: "#ffffff", alpha: 1, radiusScale: 1, ...over,
});

const frame = (users: DrawableUser[], over: Partial<DotsFrame> = {}): DotsFrame => ({
  users, alphaScale: 1, radiusScale: 1, ...over,
});

describe("drawDots", () => {
  it("clears the graphics on every call", () => {
    const m = mockGraphics();
    drawDots(m.g, frame([]), { x: 0, y: 0, scale: 1 });
    expect(m.clears.count).toBe(1);
    expect(m.circles).toHaveLength(0);
  });

  it("draws one circle per user in the frame", () => {
    const m = mockGraphics();
    drawDots(m.g, frame([u({ id: 0 }), u({ id: 1 }), u({ id: 2 })]), { x: 0, y: 0, scale: 1 });
    expect(m.circles).toHaveLength(3);
  });

  it("places circles at world coordinates from the frame", () => {
    const m = mockGraphics();
    drawDots(m.g, frame([u({ x: 3, y: 4 })]), { x: 100, y: 50, scale: 2 });
    expect(m.circles[0]!.x).toBeCloseTo(3);
    expect(m.circles[0]!.y).toBeCloseTo(4);
  });

  it("scales the radius by 1/sqrt(scale) and by frame.radiusScale", () => {
    const m = mockGraphics();
    drawDots(m.g, frame([u({})], { radiusScale: 2 }), { x: 0, y: 0, scale: 4 });
    // tokens.dot.radius is 4; (4 * 2) / sqrt(4) = 4
    expect(m.circles[0]!.r).toBeCloseTo(4, 6);
  });

  it("realizes alphaScale < 1 as a post-composite group-fade filter, not per-fragment alpha", () => {
    const m = mockGraphics();
    const fade = mockFade();
    drawDots(m.g, frame([u({ alpha: 0.5 })], { alphaScale: 0.3 }), { x: 0, y: 0, scale: 1 }, fade);
    // The filter carries the group opacity (composited once, no overdraw)...
    expect(fade.alpha).toBeCloseTo(0.3, 6);
    expect(m.raw.filters).toEqual([fade]);
    // ...while the layer and every dot keep full alpha. Folding alphaScale into
    // g.alpha (worldAlpha) would multiply each fragment, so stacked dots would
    // still saturate — the exact bug this avoids.
    expect(m.raw.alpha).toBe(1);
    expect(m.fills[0]!.alpha).toBeCloseTo(0.5, 6);
  });

  it("attaches no group-fade filter when alphaScale is 1", () => {
    const m = mockGraphics();
    drawDots(m.g, frame([u({ alpha: 0.5 })], { alphaScale: 1 }), { x: 0, y: 0, scale: 1 }, mockFade());
    expect(m.raw.filters).toBeFalsy();
    expect(m.raw.alpha).toBe(1);
    expect(m.fills[0]!.alpha).toBeCloseTo(0.5, 6);
  });

  it("detaches the group-fade filter when a later frame leaves the fade window", () => {
    const m = mockGraphics();
    const fade = mockFade();
    drawDots(m.g, frame([u({})], { alphaScale: 0.3 }), { x: 0, y: 0, scale: 1 }, fade);
    expect(m.raw.filters).toEqual([fade]);
    drawDots(m.g, frame([u({})], { alphaScale: 1 }), { x: 0, y: 0, scale: 1 }, fade);
    expect(m.raw.filters).toEqual([]);
  });

  it("passes the resolved color from the frame straight through", () => {
    const m = mockGraphics();
    drawDots(m.g, frame([u({ color: "#abcdef" })]), { x: 0, y: 0, scale: 1 });
    expect(m.fills[0]!.color).toBe("#abcdef");
  });

  it("multiplies the base radius by u.radiusScale", () => {
    const m = mockGraphics();
    drawDots(m.g, frame([u({ radiusScale: 2 })]), { x: 0, y: 0, scale: 1 });
    expect(m.circles[0]!.r).toBeCloseTo(8, 6);
  });
});
