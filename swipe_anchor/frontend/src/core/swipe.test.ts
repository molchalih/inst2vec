import { describe, expect, it } from "vitest";
import { gestureConfidence, swipeGesture } from "./swipe";

const base = { dx: 0, dy: 0, vx: 0, vy: 0, width: 360, height: 720 };

describe("swipeGesture", () => {
  it("returns null for a small, slow drag", () => {
    expect(swipeGesture({ ...base, dx: -20 })).toBeNull();
  });

  it("reads a long horizontal drag as left/right", () => {
    expect(swipeGesture({ ...base, dx: -90 })).toBe("left");
    expect(swipeGesture({ ...base, dx: 90 })).toBe("right");
  });

  it("reads a long vertical drag as up/down", () => {
    expect(swipeGesture({ ...base, dy: -160 })).toBe("up");
    expect(swipeGesture({ ...base, dy: 160 })).toBe("down");
  });

  it("lets the dominant axis win", () => {
    // mostly horizontal with a little vertical wobble -> horizontal
    expect(swipeGesture({ ...base, dx: -90, dy: -30 })).toBe("left");
    // mostly vertical -> vertical
    expect(swipeGesture({ ...base, dx: -30, dy: -160 })).toBe("up");
  });

  it("commits a fast flick on a short distance", () => {
    expect(swipeGesture({ ...base, dx: -8, vx: -0.8 })).toBe("left");
    expect(swipeGesture({ ...base, dy: -8, vy: -0.8 })).toBe("up");
  });

  it("honours custom thresholds", () => {
    expect(swipeGesture({ ...base, dx: -40 }, { distanceFrac: 0.05 })).toBe("left");
    expect(swipeGesture({ ...base, dx: -40 }, { distanceFrac: 0.5 })).toBeNull();
    expect(swipeGesture({ ...base, vx: 0.25 }, { flickVelocity: 0.2 })).toBe("right");
    expect(swipeGesture({ ...base, vx: 0.25 }, { flickVelocity: 0.9 })).toBeNull();
  });
});

describe("gestureConfidence", () => {
  it("is low for a slow gesture and saturates for a fast one", () => {
    expect(gestureConfidence(0)).toBeLessThan(0.2);
    expect(gestureConfidence(2.0)).toBe(1);
    expect(gestureConfidence(-2.0)).toBe(1);
  });
});
