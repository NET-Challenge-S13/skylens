// Real-world coordinates. The app operates in a local ENU (East/North/Up) meter
// frame anchored at a reference GPS point, and the Three.js scene maps 1 unit = 1
// meter. This keeps drone routes, telemetry, detections, and splat alignment all
// expressible in real GPS while the renderer stays in a simple metric frame.
//
// Kept Three-free (skylens_core is pure): scene coords are returned as Vec3 tuples.

import type { Vec3 } from './types.ts';

export interface Gps {
  lat: number;
  lon: number;
  /** Altitude in meters (ellipsoidal/relative — used as-is for Up). */
  alt: number;
}

/** Reference point that defines the local ENU origin. */
export type GeoAnchor = Gps;

export interface Enu {
  e: number;
  n: number;
  u: number;
}

const R = 6378137; // Earth radius (m)
const DEG = Math.PI / 180;

/** GPS -> local ENU meters (equirectangular small-area approximation). */
export function gpsToEnu(gps: Gps, anchor: GeoAnchor): Enu {
  const dLat = (gps.lat - anchor.lat) * DEG;
  const dLon = (gps.lon - anchor.lon) * DEG;
  return {
    e: dLon * R * Math.cos(anchor.lat * DEG),
    n: dLat * R,
    u: gps.alt - anchor.alt,
  };
}

/** Local ENU meters -> GPS (inverse of gpsToEnu). */
export function enuToGps(enu: Enu, anchor: GeoAnchor): Gps {
  return {
    lat: anchor.lat + (enu.n / R) / DEG,
    lon: anchor.lon + (enu.e / (R * Math.cos(anchor.lat * DEG))) / DEG,
    alt: anchor.alt + enu.u,
  };
}

// Scene convention (Three.js, right-handed): x = East, y = Up, z = -North.
export function enuToScene(enu: Enu): Vec3 {
  return [enu.e, enu.u, -enu.n];
}

export function sceneToEnu(v: Vec3): Enu {
  return { e: v[0], u: v[1], n: -v[2] };
}

export function gpsToScene(gps: Gps, anchor: GeoAnchor): Vec3 {
  return enuToScene(gpsToEnu(gps, anchor));
}

export function sceneToGps(v: Vec3, anchor: GeoAnchor): Gps {
  return enuToGps(sceneToEnu(v), anchor);
}
