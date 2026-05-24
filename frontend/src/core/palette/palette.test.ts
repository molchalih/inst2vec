import { describe, expect, it } from "vitest";
import { colorForCluster } from "./palette";

const palette = ["#aaaaaa", "#bbbbbb", "#cccccc"] as const;
const noise = "#000000";

describe("colorForCluster", () => {
  it("returns the same color for the same id (deterministic)", () => {
    expect(colorForCluster(2, palette, noise)).toBe(colorForCluster(2, palette, noise));
  });
  it("cycles through the palette by modulo", () => {
    expect(colorForCluster(0, palette, noise)).toBe(palette[0]);
    expect(colorForCluster(palette.length, palette, noise)).toBe(palette[0]);
  });
  it("returns the noise color for cluster id -1", () => {
    expect(colorForCluster(-1, palette, noise)).toBe(noise);
  });
});
