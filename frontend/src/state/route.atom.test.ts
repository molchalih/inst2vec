import { describe, expect, it } from "vitest";
import { parseHash, serializeRoute } from "./route.atom";

describe("route hash", () => {
  it("parses cluster=", () => {
    expect(parseHash("#cluster=7")).toEqual({ cluster: 7 });
  });

  it("parses user=", () => {
    expect(parseHash("#user=42")).toEqual({ user: 42 });
  });

  it("user wins when both present (deferred to writer to avoid producing this state)", () => {
    expect(parseHash("#cluster=7&user=42")).toEqual({ cluster: 7, user: 42 });
  });

  it("serialises cluster=", () => {
    expect(serializeRoute({ cluster: 7 })).toBe("#cluster=7");
  });

  it("serialises user= without cluster=", () => {
    expect(serializeRoute({ user: 42 })).toBe("#user=42");
  });

  it("legacy `selected=` is ignored silently", () => {
    expect(parseHash("#selected=7")).toEqual({});
  });
});
