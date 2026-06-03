import { describe, it, expect } from "vitest";
import {
  groundingLabel,
  TAG_KIND_ORDER,
  MAX_TAGS_PER_CATEGORY,
} from "./clip-label";

describe("clip-label formatting", () => {
  it("joins grounded_in with commas, prefixed by 'grounded in:'", () => {
    expect(groundingLabel(["warm kitchen", "shallow depth of field"]))
      .toBe("grounded in: warm kitchen, shallow depth of field");
  });

  it("returns empty string when grounded_in is empty", () => {
    expect(groundingLabel([])).toBe("");
  });

  it("orders kinds observable → aesthetic → community", () => {
    expect(TAG_KIND_ORDER).toEqual(["observable", "aesthetic", "community"]);
  });

  it("caps each descriptive category at five tags", () => {
    expect(MAX_TAGS_PER_CATEGORY).toBe(5);
  });
});
