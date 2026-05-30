import type { EmbeddingCase } from "@/data";

export const config = {
  baseUrl: import.meta.env.BASE_URL,
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL as string | undefined,
  cases: [
    "video",
    "sandwich",
    "auditory",
    "spoken",
    "textual",
  ] as ReadonlyArray<EmbeddingCase>,
} as const;

export type AppConfig = typeof config;
