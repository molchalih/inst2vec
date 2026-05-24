export type Palette = ReadonlyArray<string>;

export const colorForCluster = (
  id: number,
  palette: Palette,
  noise: string,
): string => (id < 0 ? noise : palette[id % palette.length]!);
