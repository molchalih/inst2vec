import { describe, expect, it } from "vitest";
import { applyWheel, applyDrag } from "./pan-zoom";

describe("applyWheel", () => {
  it("anchors zoom on the cursor — world point under cursor stays under cursor", () => {
    const before = { x: 100, y: 50, scale: 1 };
    const cursorScreen = { x: 200, y: 150 };
    const worldBefore = {
      x: (cursorScreen.x - before.x) / before.scale,
      y: (cursorScreen.y - before.y) / before.scale,
    };
    const after = applyWheel(before, cursorScreen, 1.25);
    const worldAfter = {
      x: (cursorScreen.x - after.x) / after.scale,
      y: (cursorScreen.y - after.y) / after.scale,
    };
    expect(worldAfter.x).toBeCloseTo(worldBefore.x);
    expect(worldAfter.y).toBeCloseTo(worldBefore.y);
    expect(after.scale).toBeCloseTo(1.25);
  });
});

describe("applyWheel with scale bounds", () => {
  const cursor = { x: 200, y: 150 };
  const before = { x: 100, y: 50, scale: 1 };

  it("preserves the cursor anchor when the scale is clamped to the max", () => {
    const worldBefore = {
      x: (cursor.x - before.x) / before.scale,
      y: (cursor.y - before.y) / before.scale,
    };
    const after = applyWheel(before, cursor, 10, { min: 0.5, max: 2 });
    expect(after.scale).toBeCloseTo(2);
    const worldAfter = {
      x: (cursor.x - after.x) / after.scale,
      y: (cursor.y - after.y) / after.scale,
    };
    expect(worldAfter.x).toBeCloseTo(worldBefore.x);
    expect(worldAfter.y).toBeCloseTo(worldBefore.y);
  });

  it("preserves the cursor anchor when the scale is clamped to the min", () => {
    const worldBefore = {
      x: (cursor.x - before.x) / before.scale,
      y: (cursor.y - before.y) / before.scale,
    };
    const after = applyWheel(before, cursor, 0.01, { min: 0.5, max: 2 });
    expect(after.scale).toBeCloseTo(0.5);
    const worldAfter = {
      x: (cursor.x - after.x) / after.scale,
      y: (cursor.y - after.y) / after.scale,
    };
    expect(worldAfter.x).toBeCloseTo(worldBefore.x);
    expect(worldAfter.y).toBeCloseTo(worldBefore.y);
  });

  it("is a no-op when already at the ceiling and zooming further in", () => {
    const atCeiling = { x: 100, y: 50, scale: 2 };
    const after = applyWheel(atCeiling, cursor, 2, { min: 0.5, max: 2 });
    expect(after).toEqual(atCeiling);
  });

  it("is a no-op when already at the floor and zooming further out", () => {
    const atFloor = { x: 100, y: 50, scale: 0.5 };
    const after = applyWheel(atFloor, cursor, 0.5, { min: 0.5, max: 2 });
    expect(after).toEqual(atFloor);
  });
});

describe("applyDrag", () => {
  it("translates by delta", () => {
    expect(applyDrag({ x: 10, y: 20, scale: 2 }, { x: 5, y: -3 }))
      .toEqual({ x: 15, y: 17, scale: 2 });
  });
});
