import { describe, it, expect } from "vitest";
import {
  groundingLabel,
  warningLabel,
  TAG_KIND_ORDER,
} from "./clip-label";

describe("clip-label formatting", () => {
  it("joins grounded_in with commas, prefixed by 'grounded in:'", () => {
    expect(groundingLabel(["warm kitchen", "shallow depth of field"]))
      .toBe("grounded in: warm kitchen, shallow depth of field");
  });

  it("returns empty string when grounded_in is empty", () => {
    expect(groundingLabel([])).toBe("");
  });

  it("returns the known-warning string for mapped codes", () => {
    expect(warningLabel("tag_count_out_of_range"))
      .toBe("tag count out of range");
  });

  it("falls back to the raw code when unknown", () => {
    expect(warningLabel("X9")).toBe("X9");
  });

  it("orders kinds observable → aesthetic → community", () => {
    expect(TAG_KIND_ORDER).toEqual(["observable", "aesthetic", "community"]);
  });
});
