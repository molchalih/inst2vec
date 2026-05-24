import { describe, expect, it } from "vitest";
import { worldToScreen, screenToWorld } from "./transform";

describe("transform", () => {
  const t = { x: 100, y: 50, scale: 2 };

  it("worldToScreen applies translate then scale", () => {
    expect(worldToScreen({ x: 3, y: 4 }, t)).toEqual({ x: 106, y: 58 });
  });

  it("screenToWorld is the inverse of worldToScreen", () => {
    const w = { x: -1.5, y: 7.25 };
    const s = worldToScreen(w, t);
    const w2 = screenToWorld(s, t);
    expect(w2.x).toBeCloseTo(w.x);
    expect(w2.y).toBeCloseTo(w.y);
  });
});
