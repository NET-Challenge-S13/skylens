// Real-world drone routes for the control-tower CONTROL. Unlike paths.ts (demo
// auto-sweep over the loaded scene's bounds), these are GPS routes the
// operator plans by hand and assigns to the leader drone via the route modal.

import { CONFIG } from '../config.ts';
import { gpsToScene } from '../../geo.ts';
import type { Gps } from '../../geo.ts';
import type { DronePath, Waypoint } from '../types';

const ZONE_NAMES = ['리더 (조종)', '군집 #2', '군집 #3'];

/** Default duration (seconds) to fly an assigned GPS route, unless overridden. */
export const DEFAULT_ROUTE_DURATION = 40;

/**
 * Convert an operator-authored GPS waypoint list into a scene-space
 * Catmull-Rom drone path (see math.ts samplePath), timed evenly over
 * `durationSec`. Look direction follows the heading to the next waypoint,
 * matching the convention used by paths.ts's auto-sweep.
 */
export function buildRouteFromGps(
  waypoints: Gps[],
  durationSec: number = DEFAULT_ROUTE_DURATION,
  id = 1,
  zone = ZONE_NAMES[0],
): DronePath {
  const n = waypoints.length;
  const scenePts = waypoints.map((g) => gpsToScene(g, CONFIG.geo.anchor));

  const wps: Waypoint[] = scenePts.map((pos, i) => {
    const t = n > 1 ? (i / (n - 1)) * durationSec : 0;
    const next = scenePts[Math.min(n - 1, i + 1)];
    const dx = next[0] - pos[0];
    const dz = next[2] - pos[2];
    const len = Math.hypot(dx, dz) || 1;
    return { t, pos, look: [dx / len, -0.3, dz / len] };
  });

  return { id, zone, waypoints: wps };
}

/**
 * Idle hover paths for real (non-demo) mode: each drone gets a single-point
 * "path" so samplePath holds it in place (duration 0 -> pathTime never
 * advances) until the operator assigns a real route via the modal.
 */
export function buildIdlePaths(): DronePath[] {
  return ZONE_NAMES.map((zone, i) => ({
    id: i + 1,
    zone,
    waypoints: [{ t: 0, pos: [0, 20, 0], look: [0, -1, 0.3] }],
  }));
}
