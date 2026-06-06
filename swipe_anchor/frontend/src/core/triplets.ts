/**
 * Odd-one-out -> triplet derivation (plan §1.2, §8.2 core).
 *
 * Mirrors the backend rule exactly so the client can preview / validate the two
 * ordinal constraints a single answer produces. Pure and framework-free.
 */
export interface Triplet {
  anchor: number;
  positive: number;
  negative: number;
}

export function deriveTriplets(
  creators: readonly [number, number, number],
  oddId: number,
): Triplet[] {
  if (new Set(creators).size !== 3) {
    throw new Error(`triple must hold three distinct creators: ${creators.join(",")}`);
  }
  if (!creators.includes(oddId)) {
    throw new Error(`odd id ${oddId} is not in triple ${creators.join(",")}`);
  }
  const [a, b] = creators.filter((c) => c !== oddId);
  return [
    { anchor: a!, positive: b!, negative: oddId },
    { anchor: b!, positive: a!, negative: oddId },
  ];
}
