export const formatCompact = (n: number): string => {
  const sign = n < 0 ? "-" : "";
  const abs = Math.abs(n);
  if (abs < 1000) return `${sign}${abs}`;
  if (abs < 1_000_000) return `${sign}${trim(abs / 1000)}k`;
  return `${sign}${trim(abs / 1_000_000)}M`;
};

const trim = (n: number): string => {
  // 1.0 → "1", 1.5 → "1.5", 18.4 → "18.4"
  const rounded = Math.round(n * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
};
