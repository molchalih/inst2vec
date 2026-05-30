import { describe, expect, it } from "vitest";
import { makeSources } from "./makeSources";
import { StaticBulkSource } from "./bulk/StaticBulkSource";
import { HttpBulkSource } from "./bulk/HttpBulkSource";
import { StaticApiClient } from "./api/StaticApiClient";
import { HttpApiClient } from "./api/HttpApiClient";

const getRun = () => "video";

describe("makeSources", () => {
  it("defaults to static sources when apiBaseUrl is unset", () => {
    const { bulk, api } = makeSources({ baseUrl: "/" }, getRun);
    expect(bulk).toBeInstanceOf(StaticBulkSource);
    expect(api).toBeInstanceOf(StaticApiClient);
  });

  it("flips both planes to HTTP when apiBaseUrl is set", () => {
    const { bulk, api } = makeSources(
      { baseUrl: "/", apiBaseUrl: "https://api.example.com/" },
      getRun,
    );
    expect(bulk).toBeInstanceOf(HttpBulkSource);
    expect(api).toBeInstanceOf(HttpApiClient);
  });

  it("empty-string apiBaseUrl stays static (treated as unset)", () => {
    const { bulk, api } = makeSources({ baseUrl: "/", apiBaseUrl: "" }, getRun);
    expect(bulk).toBeInstanceOf(StaticBulkSource);
    expect(api).toBeInstanceOf(StaticApiClient);
  });
});
