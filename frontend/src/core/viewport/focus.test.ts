import { describe, expect, it } from "vitest";
import { focusTransform } from "./focus";
import { fitBoundsToRect } from "../geom/fitBoundsToRect";

// Reproduction of the real bug: maest/sandwich-style narrow window where an
// edge cluster's natural fit zoom exceeds the clamp's scale ceiling. The
// stretched run bounds are the viewport box centered on the origin.
const W = 998;
const H = 1024;
const runBounds = { minX: -W / 2, maxX: W / 2, minY: -H / 2, maxY: H / 2 };
// Visible rect: inspector panel (372px) occludes the left.
const rect = { x: 372, y: 0, width: W - 372, height: H };
const limits = { minScaleFactor: 1, maxScaleFactor: 5, panMarginPx: 750 };
// fit of the whole run into the visible rect (runFitPadding 0.1).
const fitScale = fitBoundsToRect(runBounds, rect, 0.1).scale;

describe("focusTransform — scale band", () => {
  it("caps the zoom at maxScaleFactor × fitScale when the cluster fit wants more", () => {
    const desired = fitScale * 9; // far above the 5× ceiling
    const t = focusTransform({ x: 0, y: 0 }, rect, desired, fitScale, limits);
    expect(t.scale).toBeCloseTo(limits.maxScaleFactor * fitScale);
  });

  it("raises the zoom to minScaleFactor × fitScale when the cluster fit wants less", () => {
    const desired = fitScale * 0.2;
    const t = focusTransform({ x: 0, y: 0 }, rect, desired, fitScale, limits);
    expect(t.scale).toBeCloseTo(limits.minScaleFactor * fitScale);
  });

  it("passes a desired scale inside the band through unchanged", () => {
    const desired = fitScale * 3;
    const t = focusTransform({ x: 0, y: 0 }, rect, desired, fitScale, limits);
    expect(t.scale).toBeCloseTo(desired);
  });
});

describe("focusTransform — centering", () => {
  it("lands the focused point at the visible-rect center (not behind the panel)", () => {
    // An edge cluster: center sits near the left edge of the stretched run.
    const center = { x: -452, y: -424 };
    const desired = fitScale * 9; // would be clamped to the 5× ceiling
    const t = focusTransform(center, rect, desired, fitScale, limits);

    const screenX = t.x + center.x * t.scale;
    const screenY = t.y + center.y * t.scale;
    expect(screenX).toBeCloseTo(rect.x + rect.width / 2); // 685 — visible-rect center
    expect(screenY).toBeCloseTo(rect.y + rect.height / 2);
    // The panel covers x < 372; the focused point must be well clear of it.
    expect(screenX).toBeGreaterThan(372);
  });
});
