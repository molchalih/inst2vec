import { describe, expect, it } from "vitest";
import { isCreatorInRun, type CreatorPresenceRun } from "./presence";

const run = (ids: number[]): CreatorPresenceRun => ({
  users: ids.map((id) => [id, 0, 0, 0, false, 0]),
});

describe("isCreatorInRun", () => {
  it("returns true when the creator id is present", () => {
    expect(isCreatorInRun(run([10, 20, 30]), 20)).toBe(true);
  });

  it("returns false when the creator id is absent", () => {
    expect(isCreatorInRun(run([10, 20, 30]), 99)).toBe(false);
  });

  it("returns false for a null run", () => {
    expect(isCreatorInRun(null, 20)).toBe(false);
  });

  it("returns false for an empty users array", () => {
    expect(isCreatorInRun(run([]), 20)).toBe(false);
  });
});
