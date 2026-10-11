// Wire protocol for CONTROL -> STATUS state streaming over the WebRTC DataChannel.
//
// Data flow is one-directional: the CONTROL computer owns the simulation (drone
// poses, the visited buffer, active drone, sim clock) and streams snapshots to
// the STATUS computer, which computes reveal + camera + detections locally from
// what it receives. Confirmations happen on STATUS and stay local, so nothing
// needs to travel back.

import * as THREE from 'three';
import type { AppState, DroneMode, Vec3 } from './types';
import type { Gps } from '../geo.ts';

/** One drone's pose on the wire (plain arrays, no THREE objects). */
export interface WireDrone {
  id: number;
  zone: string;
  pos: Vec3;
  quat: [number, number, number, number];
  forward: Vec3;
  mode: DroneMode;
  pathTime: number;
}

export interface WireVisited {
  pos: Vec3;
  t: number;
}

/** A full CONTROL->STATUS snapshot. `visitedDelta` carries only new visited spots. */
export interface StateSnapshot {
  kind: 'state';
  time: number;
  activeDroneId: number;
  drones: WireDrone[];
  visitedDelta: WireVisited[];
  /** Running total of visited entries on the sender, so STATUS can detect gaps. */
  visitedTotal: number;
}

/**
 * Encode CONTROL's current state into a snapshot. `sinceVisited` is the number
 * of visited entries already sent, so we transmit only the delta.
 */
export function encodeState(state: AppState, sinceVisited: number): StateSnapshot {
  const drones: WireDrone[] = state.drones.map((d) => ({
    id: d.id,
    zone: d.zone,
    pos: [d.pos.x, d.pos.y, d.pos.z],
    quat: [d.quat.x, d.quat.y, d.quat.z, d.quat.w],
    forward: [d.forward.x, d.forward.y, d.forward.z],
    mode: d.mode,
    pathTime: d.pathTime,
  }));

  const visitedDelta: WireVisited[] = [];
  for (let i = Math.max(0, sinceVisited); i < state.visited.length; i++) {
    const v = state.visited[i];
    visitedDelta.push({ pos: v.pos, t: v.t });
  }

  return {
    kind: 'state',
    time: state.time,
    activeDroneId: state.activeDroneId,
    drones,
    visitedDelta,
    visitedTotal: state.visited.length,
  };
}

/**
 * Apply a received snapshot onto STATUS's local store state, mutating in place.
 * Reuses existing DroneRuntime objects/vectors where possible to avoid churn.
 */
export function applyState(snap: StateSnapshot, state: AppState): void {
  state.time = snap.time;
  state.activeDroneId = snap.activeDroneId;

  // Sync drones by id.
  for (const wd of snap.drones) {
    let d = state.drones.find((x) => x.id === wd.id);
    if (!d) {
      d = {
        id: wd.id,
        zone: wd.zone,
        pos: new THREE.Vector3(),
        quat: new THREE.Quaternion(),
        forward: new THREE.Vector3(),
        mode: wd.mode,
        pathTime: wd.pathTime,
      };
      state.drones.push(d);
    }
    d.zone = wd.zone;
    d.pos.set(wd.pos[0], wd.pos[1], wd.pos[2]);
    d.quat.set(wd.quat[0], wd.quat[1], wd.quat[2], wd.quat[3]);
    d.forward.set(wd.forward[0], wd.forward[1], wd.forward[2]);
    d.mode = wd.mode;
    d.pathTime = wd.pathTime;
  }

  // Append visited delta. If the sender is ahead of us by more than the delta
  // (e.g. we joined late), we just take what we're given — reveal is monotonic
  // so missing early spots only means a slightly later bloom, never corruption.
  for (const v of snap.visitedDelta) {
    state.visited.push({ pos: v.pos, t: v.t });
  }
}

// ---------------------------------------------------------------------------
// Server messages (server -> client). In real operation these come from the
// backend (detection model, splat reconstructor, drone telemetry). With no live
// backend yet they are produced by a mock provider (see server/serverSource.ts).
// ---------------------------------------------------------------------------

/** Where a splat chunk lands in the shared ENU/scene frame. */
export interface SplatAlign {
  /** Optional GPS anchor to place the chunk at; else it uses the scene origin. */
  anchor?: Gps;
  /** Explicit transform (scene units), applied after any anchor placement. */
  position: Vec3;
  rotation: [number, number, number, number]; // quaternion
  scale: [number, number, number];
}

/**
 * One reconstructed piece of the scene. The stream follows the DELAY PATTERN
 * (interim report Ⅱ-3-다): the flight is cut into segments, each segment is
 * delivered at a low training-step level first and refined afterwards, and the
 * refinement of one segment overlaps the first delivery of the next.
 *
 * A higher `level` for a `segment` REPLACES the lower one already on the board;
 * different segments accumulate side by side.
 */
export interface SplatChunk {
  kind: 'splat-chunk';
  id: string;
  url: string;
  align: SplatAlign;
  /** Which capture segment this reconstructs (a piece of the scene, not a copy). */
  segment: number;
  /** Refinement level inside the segment; 1 lands first, higher ones replace it. */
  level: number;
  /** Training steps behind this level (report 표 8). 0 when unknown. */
  steps: number;
  /** What the commander can make out at this level (report 표 8). */
  label: string;
  /** True when no further level will arrive for this segment. */
  final: boolean;
}

/** Refinement state of one segment, as the board currently holds it. */
export interface SegmentStatus {
  index: number;
  /** Highest level delivered so far; 0 = still being processed. */
  level: number;
  /** Levels this segment will go through in total. */
  levels: number;
  /** Training steps behind the delivered level. */
  steps: number;
  label: string;
}

export interface DroneTelemetry {
  kind: 'telemetry';
  id: number;
  gps: Gps;
  headingDeg: number;
}

export interface DetectionResult {
  kind: 'detection';
  id: string;
  category: 'person' | 'danger';
  gps: Gps;
  confidence: number;
  label: string;
}

export interface ServerStatus {
  kind: 'server-status';
  connected: boolean;
  receiving: boolean;
  chunks: number;
  detections: number;
  lastSeq: number;
  latencyMs: number | null;
  /** Per-segment refinement state; empty when the source streams a single scene. */
  segments: SegmentStatus[];
}

/** client -> server: assign a GPS route to a drone. */
export interface AssignRoute {
  kind: 'assign-route';
  droneId: number;
  waypoints: Gps[];
}

export type ServerMessage = SplatChunk | DroneTelemetry | DetectionResult | ServerStatus;

export const IDENTITY_ALIGN: SplatAlign = {
  position: [0, 0, 0],
  rotation: [0, 0, 0, 1],
  scale: [1, 1, 1],
};
