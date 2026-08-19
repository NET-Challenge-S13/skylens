// The fleet as the CORE reports it (COMPONENTS.md §8: 관제탑↔현황판 P2P 폐기).
//
// This replaces the old pathFollower authority. The tower used to own the
// simulation — it advanced drones along preset paths and streamed the result to
// the situation board — which made the operator screen the source of truth for
// where drones are. It is not. The core is. So this module holds NO motion
// model: it takes `DroneTelemetry` and renders it.
//
// The only local computation is SMOOTHING. Telemetry lands at a handful of Hz
// and the viewer draws at 60, so positions are eased toward the last reported
// GPS instead of stepping. That is interpolation between known truths, never
// extrapolation past them: when telemetry stops, the drone stops, and the link
// panel says why. It never keeps flying on its own.

import * as THREE from 'three';
import { state } from '../../shared/viewer/store.ts';
import { CONFIG, DRONE_TINTS } from '../../shared/viewer/config.ts';
import { dampFactor } from '../../shared/viewer/math.ts';
import type { DroneTelemetry } from '../../shared/protocol.ts';
import type { Gps } from '../../shared/geo.ts';
import type { DroneRuntime } from '../../shared/viewer/types.ts';
import type { GeoFrame } from '../geoFrame.ts';

/** Seconds without telemetry before a drone is shown as stale. */
const STALE_AFTER = 3;
/** Smoothing rate; higher converges faster on the reported position. */
const POSITION_DAMP = 5;
const HEADING_DAMP = 6;

export interface FleetDrone {
  id: number;
  /** Last reported GPS — the tower's own idea of "where it is". */
  gps: Gps;
  headingDeg: number;
  speed: number;
  batteryPct: number;
  /** Unix ms the drone stamped the reading with. */
  reportedAt: number;
  /** Unix ms the tower received it — the difference is one-way link lag. */
  receivedAt: number;
  stale: boolean;
}

export interface TelemetryFleet {
  /** Feed one telemetry frame straight off the core link. */
  ingest(t: DroneTelemetry): void;
  /** Advance smoothing. Call every frame. */
  update(dt: number): void;
  /** Snapshot for the readout panels, ordered by drone id. */
  drones(): FleetDrone[];
  get(id: number): FleetDrone | undefined;
  /** True when nothing has ever been received — the "no fleet" screen state. */
  readonly empty: boolean;
  /** Forget everything (core link lost for good / operator reset). */
  clear(): void;
}

interface Track {
  info: FleetDrone;
  runtime: DroneRuntime;
  /** Where smoothing is heading, in world units. */
  target: THREE.Vector3;
  targetForward: THREE.Vector3;
  /** No frame yet → snap instead of easing in from the origin. */
  placed: boolean;
}

/** Compass heading (0°=North, clockwise) → scene forward (North is −Z). */
function headingToForward(deg: number, out = new THREE.Vector3()): THREE.Vector3 {
  const r = (deg * Math.PI) / 180;
  return out.set(Math.sin(r), 0, -Math.cos(r)).normalize();
}

function quatFromForward(forward: THREE.Vector3): THREE.Quaternion {
  const m = new THREE.Matrix4();
  m.lookAt(new THREE.Vector3(0, 0, 0), forward, new THREE.Vector3(0, 1, 0));
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

export function createTelemetryFleet(frame: GeoFrame): TelemetryFleet {
  const tracks = new Map<number, Track>();

  const zoneFor = (id: number): string =>
    id === 1 ? '리더 (조종)' : `군집 #${id}`;

  function ensure(t: DroneTelemetry): Track {
    let track = tracks.get(t.droneId);
    if (track) return track;

    const runtime: DroneRuntime = {
      id: t.droneId,
      zone: zoneFor(t.droneId),
      pos: new THREE.Vector3(),
      quat: new THREE.Quaternion(),
      forward: new THREE.Vector3(0, 0, 1),
      // The core flies the drone; from the tower's side it is always AUTO
      // unless the operator is actively holding a manual key (set elsewhere).
      mode: 'AUTO',
      pathTime: 0,
    };
    track = {
      info: {
        id: t.droneId,
        gps: { ...t.gps },
        headingDeg: t.headingDeg,
        speed: t.speed,
        batteryPct: t.batteryPct,
        reportedAt: t.t,
        receivedAt: Date.now(),
        stale: false,
      },
      runtime,
      target: new THREE.Vector3(),
      targetForward: new THREE.Vector3(0, 0, 1),
      placed: false,
    };
    tracks.set(t.droneId, track);

    state.drones = [...state.drones, runtime].sort((a, b) => a.id - b.id);
    if (!state.drones.some((d) => d.id === state.activeDroneId)) {
      state.activeDroneId = state.drones[0]?.id ?? 1;
    }
    return track;
  }

  return {
    ingest(t: DroneTelemetry): void {
      const track = ensure(t);
      track.info.gps = { ...t.gps };
      track.info.headingDeg = t.headingDeg;
      track.info.speed = t.speed;
      track.info.batteryPct = t.batteryPct;
      track.info.reportedAt = t.t;
      track.info.receivedAt = Date.now();
      track.info.stale = false;

      // THE conversion boundary: GPS in, world units out. Nothing above this
      // line has ever seen a scene coordinate.
      frame.toScene(t.gps, track.target);
      // Keep the drone above the terrain it is flying over even if the reported
      // altitude datum disagrees slightly with the DEM — a drone rendered
      // underground reads as a bug, not as data.
      const ground = frame.groundYAt(t.gps);
      if (track.target.y < ground) track.target.y = ground;
      headingToForward(t.headingDeg, track.targetForward);

      if (!track.placed) {
        track.placed = true;
        track.runtime.pos.copy(track.target);
        track.runtime.forward.copy(track.targetForward);
        track.runtime.quat.copy(quatFromForward(track.targetForward));
      }
    },

    update(dt: number): void {
      const now = Date.now();
      for (const track of tracks.values()) {
        track.info.stale = (now - track.info.receivedAt) / 1000 > STALE_AFTER;
        if (!track.placed) continue;
        // Stale telemetry: hold the last known pose. Do NOT dead-reckon.
        if (track.info.stale) continue;
        track.runtime.pos.lerp(track.target, dampFactor(POSITION_DAMP, dt));
        track.runtime.forward
          .lerp(track.targetForward, dampFactor(HEADING_DAMP, dt))
          .normalize();
        track.runtime.quat.slerp(
          quatFromForward(track.runtime.forward),
          dampFactor(HEADING_DAMP, dt),
        );
        // Visited trail feeds nothing on the tower any more (the board gates on
        // chunk arrival now), but the shared state contract still carries it.
        state.visited.push({
          pos: [track.runtime.pos.x, track.runtime.pos.y, track.runtime.pos.z],
          t: state.time,
        });
        if (state.visited.length > 4000) state.visited.splice(0, state.visited.length - 4000);
      }
    },

    drones(): FleetDrone[] {
      return [...tracks.values()].map((t) => ({ ...t.info })).sort((a, b) => a.id - b.id);
    },

    get(id: number): FleetDrone | undefined {
      const t = tracks.get(id);
      return t ? { ...t.info } : undefined;
    },

    get empty() {
      return tracks.size === 0;
    },

    clear(): void {
      tracks.clear();
      state.drones = [];
    },
  };
}

/** Palette accessor so the panels tint drone rows like the 3D rigs. */
export function droneTint(id: number): number {
  return DRONE_TINTS[(id - 1) % DRONE_TINTS.length] ?? CONFIG.color.droneCore;
}
