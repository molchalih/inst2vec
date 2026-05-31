/**
 * Pure membership predicate: is a creator present in a loaded run?
 *
 * Typed structurally (only the `users` tuples' first slot — the creator id —
 * is read) so `core/` stays framework-free and imports nothing from `src/`.
 * `AtlasRun` from `@/data` is assignable to `CreatorPresenceRun`.
 */
export type CreatorPresenceRun = {
  users: ReadonlyArray<readonly [number, ...unknown[]]>;
};

export function isCreatorInRun(
  run: CreatorPresenceRun | null,
  creatorId: number,
): boolean {
  if (run === null) return false;
  return run.users.some(([id]) => id === creatorId);
}
