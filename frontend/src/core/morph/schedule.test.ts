import { describe, expect, it } from "vitest";
import {
  ellipseAlphaScale, ellipseSide, userAlphaSchedule,
} from "./schedule";

describe("ellipseAlphaScale", () => {
  it("is 1 during phase 0", () => {
    expect(ellipseAlphaScale(0, 0)).toBe(1);
    expect(ellipseAlphaScale(0, 0.5)).toBe(1);
    expect(ellipseAlphaScale(0, 1)).toBe(1);
  });
  it("eases 1 → 0 during phase 1", () => {
    expect(ellipseAlphaScale(1, 0)).toBe(1);
    expect(ellipseAlphaScale(1, 0.5)).toBe(0.5);
    expect(ellipseAlphaScale(1, 1)).toBe(0);
  });
  it("is 0 during phase 2", () => {
    expect(ellipseAlphaScale(2, 0)).toBe(0);
    expect(ellipseAlphaScale(2, 1)).toBe(0);
  });
  it("eases 0 → 1 during phase 3", () => {
    expect(ellipseAlphaScale(3, 0)).toBe(0);
    expect(ellipseAlphaScale(3, 0.5)).toBe(0.5);
    expect(ellipseAlphaScale(3, 1)).toBe(1);
  });
});

describe("ellipseSide", () => {
  it("returns 'from' during phases 0 and 1", () => {
    expect(ellipseSide(0)).toBe("from");
    expect(ellipseSide(1)).toBe("from");
  });
  it("returns 'to' during phases 2 and 3", () => {
    expect(ellipseSide(2)).toBe("to");
    expect(ellipseSide(3)).toBe("to");
  });
});

describe("userAlphaSchedule", () => {
  it("creator present in both: alpha = 1 throughout the dot-morph", () => {
    expect(userAlphaSchedule({ inFrom: true, inTo: true, progress: 0 })).toBe(1);
    expect(userAlphaSchedule({ inFrom: true, inTo: true, progress: 0.5 })).toBe(1);
    expect(userAlphaSchedule({ inFrom: true, inTo: true, progress: 1 })).toBe(1);
  });
  it("creator only in from: alpha = 1 - progress", () => {
    expect(userAlphaSchedule({ inFrom: true, inTo: false, progress: 0 })).toBe(1);
    expect(userAlphaSchedule({ inFrom: true, inTo: false, progress: 0.5 })).toBe(0.5);
    expect(userAlphaSchedule({ inFrom: true, inTo: false, progress: 1 })).toBe(0);
  });
  it("creator only in to: alpha = progress", () => {
    expect(userAlphaSchedule({ inFrom: false, inTo: true, progress: 0 })).toBe(0);
    expect(userAlphaSchedule({ inFrom: false, inTo: true, progress: 0.5 })).toBe(0.5);
    expect(userAlphaSchedule({ inFrom: false, inTo: true, progress: 1 })).toBe(1);
  });
});
