/**
 * Prefetch-queue policy (plan §8.3): keep a small buffer of comparisons ahead
 * so swiping never blocks on the network. Pure decision; the fetch is a side
 * effect owned by state/.
 */
export function shouldRefill(queueLength: number, lowWaterMark = 2): boolean {
  return queueLength <= lowWaterMark;
}
