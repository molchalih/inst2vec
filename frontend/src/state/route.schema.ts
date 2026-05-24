import { z } from "zod";
import { embeddingCaseSchema } from "@/data";

export const routeSchema = z.object({
  case: embeddingCaseSchema.optional(),
  cluster: z.coerce.number().int().optional(),
  user: z.coerce.number().int().optional(),
});

export type Route = z.infer<typeof routeSchema>;
