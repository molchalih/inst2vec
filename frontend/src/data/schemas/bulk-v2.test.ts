import { describe, expect, it } from "vitest";
import { SCHEMA_VERSION } from "./version";
import { usersFileSchema } from "./users.schema";
import { clustersFileSchema } from "./clusters.schema";
import { manifestSchema } from "./manifest.schema";

describe("bulk schema v2", () => {
  it("SCHEMA_VERSION is 2", () => {
    expect(SCHEMA_VERSION).toBe(2);
  });

  it("users tuple carries a trailing has_detail boolean", () => {
    const parsed = usersFileSchema.parse({
      version: 2,
      run_id: "video-1",
      bounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 },
      users: [[42, 0.1, 0.2, 7, true]],
    });
    expect(parsed.users[0]).toEqual([42, 0.1, 0.2, 7, true]);
  });

  it("users tuple rejects a 4-wide v1 payload", () => {
    const r = usersFileSchema.safeParse({
      version: 2,
      run_id: "video-1",
      bounds: { minX: 0, maxX: 1, minY: 0, maxY: 1 },
      users: [[42, 0.1, 0.2, 7]],
    });
    expect(r.success).toBe(false);
  });

  it("clusters object carries has_detail", () => {
    const parsed = clustersFileSchema.parse({
      version: 2,
      run_id: "video-1",
      clusters: [
        { id: 0, label: "A", cx: 0, cy: 0, rx: 1, ry: 1, angle: 0, size: 1, has_detail: true },
      ],
    });
    expect(parsed.clusters[0]!.has_detail).toBe(true);
  });

  it("manifest run carries details_available", () => {
    const parsed = manifestSchema.parse({
      version: 2,
      default_run_id: "video-1",
      runs: [{ id: "video-1", case: "video", label: "v", size: 1, details_available: true }],
    });
    expect(parsed.runs[0]!.details_available).toBe(true);
  });
});
