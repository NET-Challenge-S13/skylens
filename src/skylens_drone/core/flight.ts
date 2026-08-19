// Route geometry and the flight model.
//
// Pure math over the shared ENU helpers — no timers, no I/O. The drone app ticks
// this; the operator UI reads it. Distances are metres along the polyline, so
// "how far has the drone flown" is directly the trigger the core needs for its
// delay-pattern segmentation (COMPONENTS.md §5.2).

import { enuToGps, gpsToEnu, type Enu, type Gps } from '../../shared/geo.ts';

export type FlightDirection = 'forward' | 'backward';

export interface Pose {
  gps: Gps;
  headingDeg: number;
}

export interface RoutePlan {
  waypoints: Gps[];
  loop: boolean;
  /** ENU origin for the whole plan — waypoint 0. */
  anchor: Gps;
  points: Enu[];
  /** Cumulative distance at each waypoint, metres. */
  cumulative: number[];
  /** One-way length of the route, metres. */
  totalM: number;
}

const TAU = Math.PI * 2;

function dist(a: Enu, b: Enu): number {
  return Math.hypot(b.e - a.e, b.n - a.n, b.u - a.u);
}

/** Bearing of an ENU delta, degrees clockwise from north, 0..360. */
export function bearingOf(from: Enu, to: Enu): number {
  const deg = (Math.atan2(to.e - from.e, to.n - from.n) * 360) / TAU;
  return (deg + 360) % 360;
}

export function planRoute(waypoints: Gps[], loop: boolean): RoutePlan {
  const anchor = waypoints[0] ?? { lat: 0, lon: 0, alt: 0 };
  const points = waypoints.map((w) => gpsToEnu(w, anchor));
  const cumulative: number[] = [0];
  for (let i = 1; i < points.length; i++) {
    cumulative.push(cumulative[i - 1] + dist(points[i - 1], points[i]));
  }
  return {
    waypoints: [...waypoints],
    loop,
    anchor,
    points,
    cumulative,
    totalM: cumulative[cumulative.length - 1] ?? 0,
  };
}

/**
 * Fold an ever-growing odometer onto the route.
 *
 * loop=true is a **ping-pong**: the drone flies to the last waypoint, turns
 * around, and comes back (AssignRoute.loop — "fly the route back and forth").
 * loop=false clamps at the last waypoint.
 */
export function foldOdometer(
  plan: RoutePlan,
  odometerM: number,
): { s: number; direction: FlightDirection; lap: number; done: boolean } {
  const total = plan.totalM;
  if (total <= 0) return { s: 0, direction: 'forward', lap: 0, done: true };
  if (!plan.loop) {
    const s = Math.min(odometerM, total);
    return { s, direction: 'forward', lap: 0, done: odometerM >= total };
  }
  const lap = Math.floor(odometerM / total);
  const rem = odometerM - lap * total;
  const forward = lap % 2 === 0;
  return { s: forward ? rem : total - rem, direction: forward ? 'forward' : 'backward', lap, done: false };
}

/** Position + heading at distance `s` metres along the one-way polyline. */
/**
 * Where a drone sits relative to the leader, in the leader's heading frame
 * (metres). The report's demo formation is one leader plus two wingmen holding
 * station behind and to either side (중간보고서 Ⅲ-1-나: 리더 드론 1대와 이를
 * 추종하는 군집 드론 2대).
 *
 * Followers fly the SAME assigned route with this offset applied rather than
 * chasing the leader's telemetry: each aircraft then needs nothing from the
 * others, which is how waypoint formation flight actually works and keeps the
 * drone component independent of the rest of the pipeline.
 */
export interface FormationSlot {
  /** Right of the heading. */
  right: number;
  /** Above. */
  up: number;
  /** Ahead; negative trails the leader. */
  forward: number;
}

export const FORMATION_SLOTS: readonly FormationSlot[] = [
  { right: 0, up: 0, forward: 0 },
  { right: -16, up: 4, forward: -12 },
  { right: 16, up: -4, forward: -12 },
];

/** Offset a leader pose into a formation slot. Heading is unchanged: wingmen
 *  hold the leader's attitude, which is what makes the group read as one. */
export function applyFormation(pose: Pose, slot: FormationSlot, anchor: Gps): Pose {
  if (slot.right === 0 && slot.up === 0 && slot.forward === 0) return pose;
  const rad = (pose.headingDeg * TAU) / 360;
  // Heading is a compass bearing: north is +n, east is +e.
  const fwd = { e: Math.sin(rad), n: Math.cos(rad) };
  const right = { e: fwd.n, n: -fwd.e };
  const here = gpsToEnu(pose.gps, anchor);
  return {
    gps: enuToGps(
      {
        e: here.e + right.e * slot.right + fwd.e * slot.forward,
        n: here.n + right.n * slot.right + fwd.n * slot.forward,
        u: here.u + slot.up,
      },
      anchor,
    ),
    headingDeg: pose.headingDeg,
  };
}

export function sampleRoute(plan: RoutePlan, s: number, direction: FlightDirection): Pose {
  const pts = plan.points;
  if (pts.length === 0) return { gps: plan.anchor, headingDeg: 0 };
  if (pts.length === 1) return { gps: plan.waypoints[0], headingDeg: 0 };

  const clamped = Math.max(0, Math.min(s, plan.totalM));
  let i = 1;
  while (i < plan.cumulative.length - 1 && plan.cumulative[i] < clamped) i++;
  const segStart = plan.cumulative[i - 1];
  const segLen = plan.cumulative[i] - segStart;
  const t = segLen > 0 ? (clamped - segStart) / segLen : 0;
  const a = pts[i - 1];
  const b = pts[i];
  const here: Enu = {
    e: a.e + (b.e - a.e) * t,
    n: a.n + (b.n - a.n) * t,
    u: a.u + (b.u - a.u) * t,
  };
  const heading = direction === 'forward' ? bearingOf(a, b) : bearingOf(b, a);
  return { gps: enuToGps(here, plan.anchor), headingDeg: heading };
}

/**
 * Where the drone parks before it is dispatched. Real hardware knows its own
 * base; in demo we place it `offsetM` south-west of waypoint 0 and lower, so the
 * transit leg is visible on the map as a genuine approach.
 */
export function deriveHome(plan: RoutePlan, offsetM: number): Gps {
  const start = plan.points[0] ?? { e: 0, n: 0, u: 0 };
  const next = plan.points[1] ?? { e: start.e + 1, n: start.n, u: start.u };
  const away = bearingOf(next, start) * (TAU / 360);
  return enuToGps(
    {
      e: start.e + Math.sin(away) * offsetM,
      n: start.n + Math.cos(away) * offsetM,
      u: start.u - 25,
    },
    plan.anchor,
  );
}

/** Straight-line interpolation used for the base -> site transit. */
export function interpolateGps(from: Gps, to: Gps, t: number, anchor: Gps): Pose {
  const a = gpsToEnu(from, anchor);
  const b = gpsToEnu(to, anchor);
  const k = Math.max(0, Math.min(1, t));
  return {
    gps: enuToGps({ e: a.e + (b.e - a.e) * k, n: a.n + (b.n - a.n) * k, u: a.u + (b.u - a.u) * k }, anchor),
    headingDeg: bearingOf(a, b),
  };
}

export function groundDistance(from: Gps, to: Gps, anchor: Gps): number {
  return dist(gpsToEnu(from, anchor), gpsToEnu(to, anchor));
}

/**
 * Integrate one tick of operator stick input. forward/yaw/climb are -1..1
 * (ManualControl), heading is degrees, speeds are m/s and deg/s.
 */
export function stepManual(
  pose: Pose,
  input: { forward: number; yaw: number; climb: number },
  dt: number,
  opts: { speed: number; climbSpeed: number; yawRateDeg: number },
  anchor: Gps,
): Pose {
  const heading = (pose.headingDeg + input.yaw * opts.yawRateDeg * dt + 360) % 360;
  const rad = (heading * TAU) / 360;
  const step = input.forward * opts.speed * dt;
  const enu = gpsToEnu(pose.gps, anchor);
  return {
    gps: enuToGps(
      {
        e: enu.e + Math.sin(rad) * step,
        n: enu.n + Math.cos(rad) * step,
        u: enu.u + input.climb * opts.climbSpeed * dt,
      },
      anchor,
    ),
    headingDeg: heading,
  };
}
