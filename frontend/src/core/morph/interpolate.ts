import { colorForCluster, type Palette } from "../palette/palette";
import { userAlphaSchedule } from "./schedule";
import type { JoinedCluster, JoinedUser } from "./join";

export type InterpolatedUser = {
  id: number;
  x: number;
  y: number;
  color: string;
  alpha: number;
};

export type InterpolatedEllipse = {
  id: number;
  cx: number; cy: number;
  rx: number; ry: number;
  angle: number;
  color: string;
};

const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

const hexToRgb = (hex: string): [number, number, number] => {
  const m = /^#([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) throw new Error(`lerpHex: expected #rrggbb, got "${hex}"`);
  const h = m[1]!;
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
};

const rgbToHex = (r: number, g: number, b: number): string => {
  const c = (n: number) => Math.round(Math.max(0, Math.min(255, n)))
    .toString(16)
    .padStart(2, "0");
  return `#${c(r)}${c(g)}${c(b)}`;
};

export const lerpHex = (from: string, to: string, t: number): string => {
  const [fr, fg, fb] = hexToRgb(from);
  const [tr, tg, tb] = hexToRgb(to);
  return rgbToHex(lerp(fr, tr, t), lerp(fg, tg, t), lerp(fb, tb, t));
};

const colorForJoined = (
  fromCluster: number | null,
  toCluster: number | null,
  progress: number,
  palette: Palette,
  noise: string,
): string => {
  const fc = fromCluster ?? toCluster;
  const tc = toCluster ?? fromCluster;
  if (fc === null || tc === null) return noise;
  const a = colorForCluster(fc, palette, noise);
  const b = colorForCluster(tc, palette, noise);
  return lerpHex(a, b, progress);
};

export const interpolateUsers = (
  joined: ReadonlyArray<JoinedUser>,
  motionProgressFor: (index: number) => number,
  colorProgressFor: (index: number) => number,
  palette: Palette,
  noise: string,
): InterpolatedUser[] => {
  const out: InterpolatedUser[] = [];
  for (let i = 0; i < joined.length; i++) {
    const j = joined[i]!;
    const motion = motionProgressFor(i);
    const color = colorProgressFor(i);
    const fxy = j.fromXY;
    const txy = j.toXY;
    let x: number;
    let y: number;
    if (fxy && txy) {
      x = lerp(fxy[0], txy[0], motion);
      y = lerp(fxy[1], txy[1], motion);
    } else if (fxy) {
      x = fxy[0]; y = fxy[1];
    } else if (txy) {
      x = txy[0]; y = txy[1];
    } else {
      continue;
    }
    out.push({
      id: j.id,
      x, y,
      color: colorForJoined(j.fromCluster, j.toCluster, color, palette, noise),
      alpha: userAlphaSchedule({
        inFrom: fxy !== null,
        inTo: txy !== null,
        progress: motion,
      }),
    });
  }
  return out;
};

export const interpolateEllipses = (
  joined: ReadonlyArray<JoinedCluster>,
  side: "from" | "to",
  palette: Palette,
  noise: string,
): InterpolatedEllipse[] => {
  const out: InterpolatedEllipse[] = [];
  for (const j of joined) {
    const shape = side === "from" ? j.from : j.to;
    if (!shape) continue;
    out.push({
      id: j.id,
      cx: shape.cx, cy: shape.cy, rx: shape.rx, ry: shape.ry, angle: shape.angle,
      color: colorForCluster(j.id, palette, noise),
    });
  }
  return out;
};
