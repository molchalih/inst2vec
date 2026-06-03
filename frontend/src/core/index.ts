export type { Vec2 } from "./geom/vec2";
export { add, sub, scale, dot, len } from "./geom/vec2";

export type { Transform } from "./geom/transform";
export { worldToScreen, screenToWorld } from "./geom/transform";

export type { Bounds, Viewport } from "./geom/fit";
export { fitBounds } from "./geom/fit";

export type { Rect } from "./geom/fitBoundsToRect";
export { fitBoundsToRect } from "./geom/fitBoundsToRect";
export { centerWorldPointInRect } from "./geom/centerInRect";

export type { Ellipse } from "./geom/ellipse";
export { isPointInEllipse } from "./geom/ellipse";
export { ellipsePoints } from "./geom/ellipsePath";
export { stretchEllipse } from "./geom/stretch";

export type { Palette } from "./palette/palette";
export { colorForCluster } from "./palette/palette";

export type { HitTest, Dot, LabeledEllipse } from "./spatial/hit-test";
export { BruteForceHitTest } from "./spatial/hit-test";

export type { ScaleBounds } from "./viewport/pan-zoom";
export { applyWheel, applyDrag } from "./viewport/pan-zoom";

export { easeOutCubic } from "./motion/ease";
export { sinePulse } from "./motion/pulse";
export { hashUnit } from "./motion/hash";
export type { IntroPhase, IntroDurations } from "./motion/intro";
export {
  introPhaseAndProgress, introStagger, introDotAlpha, introEllipseAlpha,
} from "./motion/intro";

export type { ClampLimits, ViewportSize } from "./viewport/clamp";
export { clampPanZoom } from "./viewport/clamp";
export { focusTransform } from "./viewport/focus";

export type { JoinedUser, JoinedCluster } from "./morph/join";
export { joinUsersByCreator, joinClustersById } from "./morph/join";
export type { InterpolatedUser, InterpolatedEllipse } from "./morph/interpolate";
export { interpolateUsers, interpolateEllipses, lerpHex } from "./morph/interpolate";
export { ellipseAlphaScale, ellipseSide, userAlphaSchedule } from "./morph/schedule";
export { flightProgress, interpolatedUserPos } from "./morph/track-pos";
export {
  computeWaveDelays, waveProgress, scalePop, emergeScalePop, vanishScalePop,
} from "./morph/wave";

export { formatCompact } from "./format/compact";
export {
  groundingLabel, formatTag, TAG_KIND_ORDER, MAX_TAGS_PER_CATEGORY, type TagKind,
} from "./format/clip-label";
export { clusterSummaryLede } from "./format/cluster-label";

export { phraseFor, type Phrase } from "./distinctiveness/phrase";

export { isCreatorInRun, type CreatorPresenceRun } from "./track/presence";

export type { CentralityScaleParams } from "./dot/centrality";
export { centralityRadiusScale } from "./dot/centrality";
