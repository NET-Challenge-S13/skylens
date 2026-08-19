// Where the flight gets cut into segments.
//
// COMPONENTS.md §5.2: **딜레이 패턴의 트리거는 시간이 아니라 드론의 이동량이다.**
// There is no wall clock anywhere in this file. A segment boundary is crossed
// when the drone has MOVED, and nothing else.
//
// Arc length comes from the assigned route when there is one: the drone's GPS is
// projected onto the route polyline and the distance travelled ALONG the route is
// what buckets into segments. That matters for 데모 시나리오 5 (지정 경로 왕복
// 반복) — on the return leg the arc length runs back down, so the drone re-enters
// segments it already flew instead of inventing new ones. A segment is a place.
//
// With no route (manual stick control), the same bucketing runs on a plain
// odometer, so the scheduler downstream cannot tell the two apart.

import type { Gps } from '../../shared/geo.ts';
import { gpsToEnu } from '../../shared/geo.ts';

export interface Projection {
  /** Distance along the route from waypoint 0, meters. */
  arcM: number;
  /** Perpendicular distance from the route, meters. */
  offsetM: number;
}

interface Leg {
  ax: number;
  ay: number;
  bx: number;
  by: number;
  len: number;
  cum: number;
}

export class RouteTracker {
  private legs: Leg[] = [];
  private anchor: Gps | null = null;
  private totalM = 0;

  setRoute(waypoints: Gps[]): void {
    this.legs = [];
    this.totalM = 0;
    this.anchor = waypoints.length > 0 ? waypoints[0] : null;
    if (this.anchor === null || waypoints.length < 2) return;

    const pts = waypoints.map((w) => {
      const enu = gpsToEnu(w, this.anchor as Gps);
      return { x: enu.e, y: enu.n };
    });
    let cum = 0;
    for (let i = 0; i + 1 < pts.length; i += 1) {
      const a = pts[i];
      const b = pts[i + 1];
      const len = Math.hypot(b.x - a.x, b.y - a.y);
      if (len <= 0) continue;
      this.legs.push({ ax: a.x, ay: a.y, bx: b.x, by: b.y, len, cum });
      cum += len;
    }
    this.totalM = cum;
  }

  clear(): void {
    this.legs = [];
    this.anchor = null;
    this.totalM = 0;
  }

  get hasRoute(): boolean {
    return this.legs.length > 0;
  }

  get lengthM(): number {
    return this.totalM;
  }

  /** Nearest point on the route, as arc length. Null when no route is set. */
  project(gps: Gps): Projection | null {
    if (this.anchor === null || this.legs.length === 0) return null;
    const enu = gpsToEnu(gps, this.anchor);
    const px = enu.e;
    const py = enu.n;

    let best: Projection = { arcM: 0, offsetM: Number.POSITIVE_INFINITY };
    for (const leg of this.legs) {
      const dx = leg.bx - leg.ax;
      const dy = leg.by - leg.ay;
      let t = ((px - leg.ax) * dx + (py - leg.ay) * dy) / (leg.len * leg.len);
      if (t < 0) t = 0;
      if (t > 1) t = 1;
      const cx = leg.ax + dx * t;
      const cy = leg.ay + dy * t;
      const offsetM = Math.hypot(px - cx, py - cy);
      if (offsetM < best.offsetM) best = { arcM: leg.cum + t * leg.len, offsetM };
    }
    return best;
  }
}

/** Ground distance between two fixes, meters (altitude ignored on purpose —
 *  segments are spans of GROUND covered, and a hover-climb is not progress). */
export function groundDistanceM(a: Gps, b: Gps): number {
  const enu = gpsToEnu(b, { lat: a.lat, lon: a.lon, alt: a.alt });
  return Math.hypot(enu.e, enu.n);
}

/** Arc length -> segment index. The only place the bucket width is applied. */
export function segmentIndexFor(arcM: number, segmentMeters: number): number {
  if (!Number.isFinite(arcM) || arcM < 0) return 0;
  return Math.floor(arcM / segmentMeters);
}

export function segmentSpan(index: number, segmentMeters: number): [number, number] {
  return [index * segmentMeters, (index + 1) * segmentMeters];
}
