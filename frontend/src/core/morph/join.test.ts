import { describe, expect, it } from "vitest";
import type { CoreAtlasRun, CoreClusterShape } from "./types";
import { joinUsersByCreator, joinClustersById } from "./join";

const cluster = (over: Partial<CoreClusterShape> = {}): CoreClusterShape => ({
  id: 0, label: "c", cx: 0, cy: 0, rx: 1, ry: 1, angle: 0, size: 1, has_detail: false, ...over,
});

const run = (
  id: string,
  users: CoreAtlasRun["users"],
  clusters: CoreClusterShape[] = [],
): CoreAtlasRun => ({
  meta: { id, case: "video", label: id, size: users.length, details_available: false },
  bounds: { minX: -1, maxX: 1, minY: -1, maxY: 1 },
  users,
  clusters,
});

describe("joinUsersByCreator", () => {
  it("emits one entry per unique creator id across both runs", () => {
    const from = run("a", [[0, 0, 0, 0, false], [1, 1, 1, 0, false]]);
    const to = run("b", [[1, 2, 2, 1, false], [2, 3, 3, 1, false]]);
    const j = joinUsersByCreator(from, to);
    const ids = j.map((e) => e.id).sort((a, b) => a - b);
    expect(ids).toEqual([0, 1, 2]);
  });

  it("populates fromXY and toXY independently", () => {
    const from = run("a", [[1, 1, 1, 0, false]]);
    const to = run("b", [[1, 4, 5, 2, false]]);
    const [entry] = joinUsersByCreator(from, to);
    expect(entry!.fromXY).toEqual([1, 1]);
    expect(entry!.toXY).toEqual([4, 5]);
    expect(entry!.fromCluster).toBe(0);
    expect(entry!.toCluster).toBe(2);
  });

  it("leaves fromXY null when the creator is only in `to`", () => {
    const from = run("a", []);
    const to = run("b", [[9, 1, 2, 0, false]]);
    const [entry] = joinUsersByCreator(from, to);
    expect(entry!.fromXY).toBeNull();
    expect(entry!.toXY).toEqual([1, 2]);
  });

  it("leaves toXY null when the creator is only in `from`", () => {
    const from = run("a", [[9, 1, 2, 0, false]]);
    const to = run("b", []);
    const [entry] = joinUsersByCreator(from, to);
    expect(entry!.toXY).toBeNull();
    expect(entry!.fromXY).toEqual([1, 2]);
  });
});

describe("joinClustersById", () => {
  it("emits one entry per unique cluster id", () => {
    const from = run("a", [], [cluster({ id: 0 }), cluster({ id: 1 })]);
    const to = run("b", [], [cluster({ id: 1 }), cluster({ id: 2 })]);
    const j = joinClustersById(from, to);
    expect(j.map((e) => e.id).sort()).toEqual([0, 1, 2]);
  });

  it("excludes noise clusters (id < 0) from both sides", () => {
    const from = run("a", [], [cluster({ id: -1 }), cluster({ id: 0 })]);
    const to = run("b", [], [cluster({ id: -1 }), cluster({ id: 1 })]);
    const j = joinClustersById(from, to);
    expect(j.map((e) => e.id).sort()).toEqual([0, 1]);
  });
});
