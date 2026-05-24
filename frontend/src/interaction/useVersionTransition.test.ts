import { describe, expect, it } from "vitest";
import { phaseAndProgress } from "./useVersionTransition";

const D = { phase0: 200, phase1: 250, phase2: 600, phase3: 250 };

describe("phaseAndProgress", () => {
  it("returns phase 0 progressing during the first chunk", () => {
    const r = phaseAndProgress(0, D);
    expect(r.phase).toBe(0);
    expect(r.progress).toBeCloseTo(0);
    expect(r.done).toBe(false);
  });
  it("crosses into phase 1 at the phase-0 boundary", () => {
    const r = phaseAndProgress(200, D);
    expect(r.phase).toBe(1);
    expect(r.progress).toBeCloseTo(0);
  });
  it("crosses into phase 2 at the phase-1 boundary", () => {
    const r = phaseAndProgress(450, D);
    expect(r.phase).toBe(2);
  });
  it("crosses into phase 3 at the phase-2 boundary", () => {
    const r = phaseAndProgress(1050, D);
    expect(r.phase).toBe(3);
  });
  it("reports done at the total duration", () => {
    const r = phaseAndProgress(1300, D);
    expect(r.done).toBe(true);
  });
  it("clamps local progress to [0, 1]", () => {
    const r = phaseAndProgress(199, D);
    expect(r.progress).toBeGreaterThan(0);
    expect(r.progress).toBeLessThan(1);
  });
});
