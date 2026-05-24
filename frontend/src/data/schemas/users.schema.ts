import { z } from "zod";
import { SCHEMA_VERSION } from "./version";

export const userTupleSchema = z.tuple([
  z.number().int().nonnegative(), // id
  z.number(),                     // x
  z.number(),                     // y
  z.number().int(),               // cluster_id (-1 = noise)
  z.boolean(),                    // has_detail
]);

export const usersFileSchema = z.object({
  version: z.literal(SCHEMA_VERSION),
  run_id: z.string(),
  bounds: z.object({
    minX: z.number(),
    maxX: z.number(),
    minY: z.number(),
    maxY: z.number(),
  }),
  users: z.array(userTupleSchema),
});

export type UsersFile = z.infer<typeof usersFileSchema>;
export type UserTuple = z.infer<typeof userTupleSchema>;
