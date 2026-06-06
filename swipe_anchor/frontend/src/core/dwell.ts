/**
 * Per-card dwell accounting (plan §3 `card_dwell_ms`, §8.3). Pure: the React
 * timer that produces `deltaMs` lives in interaction/.
 */
export type DwellMap = Record<number, number>;

/** Add `deltaMs` of view time to `cardId`, returning a new map (immutable). */
export function accumulateDwell(prev: DwellMap, cardId: number, deltaMs: number): DwellMap {
  if (deltaMs <= 0) return prev;
  return { ...prev, [cardId]: (prev[cardId] ?? 0) + deltaMs };
}

/** Total dwell across all cards. */
export function sumDwell(map: DwellMap): number {
  return Object.values(map).reduce((acc, v) => acc + v, 0);
}
