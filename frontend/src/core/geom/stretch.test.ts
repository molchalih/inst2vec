import { describe, expect, it } from "vitest";
import { stretchEllipse } from "./stretch";
import type { Ellipse } from "./ellipse";

const baseEllipse = (overrides: Partial<Ellipse> = {}): Ellipse => ({
  cx: 4, cy: -2, rx: 3, ry: 1, angle: 0, ...overrides,
});

describe("stretchEllipse", () => {
  it("scales an axis-aligned ellipse by the per-axis factors and keeps angle 0", () => {
    const e = baseEllipse({ angle: 0 });
    const s = stretchEllipse(e, 2, 0.5);
    expect(s.cx).toBeCloseTo(8, 9);
    expect(s.cy).toBeCloseTo(-1, 9);
    expect(s.rx).toBeCloseTo(6, 9);
    expect(s.ry).toBeCloseTo(0.5, 9);
    expect(s.angle).toBeCloseTo(0, 9);
  });

  it("preserves angle under isotropic stretch", () => {
    const e = baseEllipse({ angle: Math.PI / 6 });
    const s = stretchEllipse(e, 3, 3);
    expect(s.angle).toBeCloseTo(Math.PI / 6, 9);
    expect(s.rx).toBeCloseTo(9, 9);
    expect(s.ry).toBeCloseTo(3, 9);
    expect(s.cx).toBeCloseTo(12, 9);
    expect(s.cy).toBeCloseTo(-6, 9);
  });

  it("rotates the angle toward the more-stretched axis under anisotropic stretch", () => {
    const e = baseEllipse({ cx: 0, cy: 0, rx: 1, ry: 1, angle: Math.PI / 4 });
    const s = stretchEllipse(e, 2, 1);
    // atan2(1 * sin(π/4), 2 * cos(π/4)) = atan2(1, 2)
    expect(s.angle).toBeCloseTo(Math.atan2(1, 2), 9);
    expect(s.rx).toBeCloseTo(2, 9);
    expect(s.ry).toBeCloseTo(1, 9);
  });
});
