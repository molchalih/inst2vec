import { describe, expect, it, vi } from "vitest";
import type { Graphics } from "pixi.js";
import { drawDots } from "./drawDots";
import type { DotsFrame, DrawableUser } from "../frame";

type CircleCall = { x: number; y: number; r: number };
type FillCall = { color: string; alpha: number };

const mockGraphics = () => {
  const circles: CircleCall[] = [];
  const fills: FillCall[] = [];
  const clears = { count: 0 };
  const g = {
    clear: vi.fn(() => { clears.count += 1; return g; }),
    circle: vi.fn((x: number, y: number, r: number) => {
      circles.push({ x, y, r }); return g;
    }),
    fill: vi.fn((opts: FillCall) => { fills.push(opts); return g; }),
  };
  return { g: g as unknown as Graphics, circles, fills, clears };
};

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

  it("multiplies per-user alpha by frame.alphaScale", () => {
    const m = mockGraphics();
    drawDots(m.g, frame([u({ alpha: 0.5 })], { alphaScale: 0.5 }), { x: 0, y: 0, scale: 1 });
    expect(m.fills[0]!.alpha).toBeCloseTo(0.25, 6);
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
