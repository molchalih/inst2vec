import { describe, expect, it } from "vitest";
import { clusterWarningLabel } from "./cluster-label";

describe("clusterWarningLabel", () => {
  it("translates known codes", () => {
    expect(clusterWarningLabel("tag_count_out_of_range")).toBe("tag count out of range");
    expect(clusterWarningLabel("invalid_confidence")).toBe("invalid confidence");
    expect(clusterWarningLabel("ungrounded_tag_reference")).toBe("ungrounded tag reference");
    expect(clusterWarningLabel("invalid_tool_tags")).toBe("invalid tool tags");
  });

  it("passes unknown codes through", () => {
    expect(clusterWarningLabel("ZZ9")).toBe("ZZ9");
  });
});
