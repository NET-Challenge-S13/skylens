// Shared types — the contract every module builds against.
import type * as THREE from 'three';

export type Vec3 = [number, number, number];

/** One key-framed point on a preset drone path (paths/drone_XX). §4.1 */
export interface Waypoint {
  /** Timecode in seconds. */
  t: number;
  pos: Vec3;
  /** Optional look-at direction (unit-ish vector). */
  look?: Vec3;
}

export interface DronePath {
  id: number;
  /** Human label / zone name, e.g. "구역 A". */
  zone: string;
  waypoints: Waypoint[];
}

/** Drone control mode state machine. §4.2 */
export type DroneMode = 'AUTO' | 'MANUAL' | 'RETURNING';

/** Live per-drone runtime state, owned by the drone module, read by viewers. */
export interface DroneRuntime {
  id: number;
  zone: string;
  /** Current world position. */
  pos: THREE.Vector3;
  /** Current orientation. */
  quat: THREE.Quaternion;
  /** Forward direction (normalized), for camera framing + models. */
  forward: THREE.Vector3;
  mode: DroneMode;
  /** Progress along preset path in seconds (the drone's own clock). */
  pathTime: number;
}

/** A spot a drone has visited — the source for progressive reveal. §5.2 */
export interface Visited {
  pos: Vec3;
  /** Sim time the spot was visited. */
  t: number;
}

/** Manually-placed detection marker (person / danger). §2-3, §5.3 */
export interface Detection {
  id: string;
  kind: 'person' | 'danger';
  pos: Vec3;
  label: string;
  confidence: number;
}

/** Runtime wrapper adding reveal/confirm lifecycle to a Detection. */
export interface DetectionRuntime extends Detection {
  /** True once the surrounding area has been revealed (viewer 2). */
  revealed: boolean;
  /** True once the operator clicked "탐지 확인". */
  confirmed: boolean;
  /** Sim time it first became revealed (for staged triggering). */
  revealedAt: number | null;
}

/** Camera sync state machine for viewer 2. §8.3 */
export type CameraSyncState =
  | 'SYNCED'
  | 'FOCUSING'
  | 'LOCKED'
  | 'RETURNING';

/** The whole shared world state. Single source of truth (§8.2 single-app). */
export interface AppState {
  /** Sim clock in seconds. */
  time: number;
  running: boolean;
  drones: DroneRuntime[];
  /** Append-only-ish list of visited spots (bounded/decimated by producer). */
  visited: Visited[];
  detections: DetectionRuntime[];
  /** Which drone keyboard input currently controls (§4.2 tab-switch). */
  activeDroneId: number;
  cameraSync: CameraSyncState;
  /** Detection currently focused/locked by viewer 2, if any. */
  focusedDetectionId: string | null;
}
