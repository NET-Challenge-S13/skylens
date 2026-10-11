// The control tower's coordinate boundary (COMPONENTS.md §8: "관제탑의 좌표는 GPS다").
//
// Everything the tower REASONS about is GPS: route waypoints from the planner,
// telemetry arriving from the core, drone placement, readouts. Nothing upstream
// of this module knows what a world unit is. Conversion happens here and only
// here, at the moment geometry is handed to Three.js.
//
// Why this is not just `shared/geo.ts`: geo.ts maps GPS to ENU METERS, and the
// tower's scene is not in meters. The VWorld terrain normalizes an arbitrary
// bbox into a fixed ~44-unit footprint and exaggerates relief, so the drone has
// to land on the SAME transform the terrain mesh and the buildings were built
// with. That transform is the TerrainContext, so the frame is derived from it —
// and geo.ts stays the pure ENU reference both stacks (TS + Python) share.

import * as THREE from 'three';
import { enuToGps, gpsToEnu } from '../shared/geo.ts';
import type { Gps } from '../shared/geo.ts';
import type { TerrainContext } from '../shared/viewer/sources/terrainSource.ts';

export interface GeoFrame {
  /** ENU origin: bbox center at the scene's lowest terrain elevation. */
  readonly anchor: Gps;
  /** Meters → world units (terrain's own normalization factor). */
  readonly unitsPerMeter: number;
  /** GPS → Three.js world position. */
  toScene(gps: Gps, out?: THREE.Vector3): THREE.Vector3;
  /** Three.js world position → GPS. */
  toGps(v: THREE.Vector3): Gps;
  /** Terrain surface height in world units under a GPS point. */
  groundYAt(gps: Gps): number;
  /** Terrain surface elevation in METERS under a GPS point. */
  groundAltAt(gps: Gps): number;
  /** Horizontal ENU meters between two GPS points. */
  metersBetween(a: Gps, b: Gps): number;
}

/**
 * Build the frame from the loaded terrain. `ctx.s` scales meters to world units
 * and `ctx.exaggeration` stretches only the vertical, so Up is NOT simply
 * `alt * s` — it is `(alt - minE) * s * exaggeration`, matching how the terrain
 * mesh and the building prisms were extruded.
 */
export function createGeoFrame(ctx: TerrainContext): GeoFrame {
  const anchor: Gps = { lat: ctx.lat0, lon: ctx.lon0, alt: ctx.minE };
  const vScale = ctx.s * ctx.exaggeration;

  const altToY = (alt: number): number => (alt - ctx.minE) * vScale;
  const yToAlt = (y: number): number => ctx.minE + y / vScale;

  return {
    anchor,
    unitsPerMeter: ctx.s,

    toScene(gps: Gps, out = new THREE.Vector3()): THREE.Vector3 {
      return out.set(
        (gps.lon - ctx.lon0) * ctx.mPerDegLon * ctx.s,
        altToY(gps.alt),
        -(gps.lat - ctx.lat0) * ctx.mPerDegLat * ctx.s,
      );
    },

    toGps(v: THREE.Vector3): Gps {
      return {
        lat: ctx.lat0 - v.z / (ctx.mPerDegLat * ctx.s),
        lon: ctx.lon0 + v.x / (ctx.mPerDegLon * ctx.s),
        alt: yToAlt(v.y),
      };
    },

    groundAltAt(gps: Gps): number {
      return ctx.elevationAt(gps.lon, gps.lat);
    },

    groundYAt(gps: Gps): number {
      return altToY(ctx.elevationAt(gps.lon, gps.lat));
    },

    metersBetween(a: Gps, b: Gps): number {
      // Pure ENU here — this is a real-world distance, not a scene measurement,
      // so it goes through the shared reference implementation.
      const ea = gpsToEnu(a, anchor);
      const eb = gpsToEnu(b, anchor);
      return Math.hypot(eb.e - ea.e, eb.n - ea.n);
    },
  };
}

/** Move a GPS point by ENU meters. Used to synthesize idle/standby positions. */
export function offsetGps(base: Gps, east: number, north: number, up = 0): Gps {
  const enu = gpsToEnu(base, base);
  return enuToGps({ e: enu.e + east, n: enu.n + north, u: enu.u + up }, base);
}
