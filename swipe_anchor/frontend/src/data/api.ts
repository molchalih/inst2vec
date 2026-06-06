import {
  nextBatchResponseSchema,
  respondResponseSchema,
  type NextBatchResponse,
  type RespondPayload,
  type RespondResponse,
} from "./schemas";

/**
 * The single network contract (plan §8.3 `data/`). `HttpApiClient` is the
 * FastAPI-backed implementation; the interface lets features/state depend on a
 * shape rather than fetch directly, and lets tests inject a fake.
 */
export interface ApiClient {
  getBatch(n: number): Promise<NextBatchResponse>;
  respond(payload: RespondPayload): Promise<RespondResponse>;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class HttpApiClient implements ApiClient {
  /**
   * @param baseUrl       backend origin
   * @param getAccessCode accessor for the per-user deeplink code; attached as the
   *                      `X-Access-Code` header on every request (the identity/auth)
   */
  constructor(
    private readonly baseUrl: string,
    private readonly getAccessCode: () => string | null = () => null,
  ) {}

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "content-type": "application/json" };
    const code = this.getAccessCode();
    if (code) h["x-access-code"] = code;
    return h;
  }

  private async post(path: string, body: unknown): Promise<unknown> {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new ApiError(`${path} failed (${res.status})`, res.status);
    }
    return res.json();
  }

  async getBatch(n: number): Promise<NextBatchResponse> {
    const raw = await this.post("/next-batch", { n });
    return nextBatchResponseSchema.parse(raw);
  }

  async respond(payload: RespondPayload): Promise<RespondResponse> {
    const raw = await this.post("/respond", payload);
    return respondResponseSchema.parse(raw);
  }
}
