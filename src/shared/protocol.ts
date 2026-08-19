// Cross-component wire contract for SkyLens (see res/docs/COMPONENTS.md).
//
//   드론 → 게이트웨이 → 프록시 → 코어 → (모델 API | 관제탑 화면 | 현황판 WebRTC)
//
// Every hop carries these same message shapes: the gateway and the proxy are
// transports, not translators. Only the model API speaks a different dialect
// (REST, see §4) because it is a request/response compute service.
//
// This module is pure data — no DOM, no Three.js, no Node built-ins — so every
// component can import it.

import type { Gps } from './geo.ts';

// ---------------------------------------------------------------------------
// §1 Identity and transport
// ---------------------------------------------------------------------------

export type ComponentId = 'drone' | 'gateway' | 'proxy' | 'core' | 'client' | 'model';

/**
 * How a drone reaches the KOREN interior.
 *   relay  — gateway forwards the media to the proxy (기본)
 *   webrtc — gateway only brokers hole punching; media goes drone ↔ proxy direct
 */
export type LinkMode = 'relay' | 'webrtc';

/**
 * Where an aircraft sits in the formation. Drones are identified to the operator
 * by their station, not by a number: "왼쪽 드론" is something you can point at on
 * the map, "드론 2" is not. The numeric id stays as the routing address.
 */
export type DroneStation = 'left' | 'center' | 'right';

export interface Envelope<T = unknown> {
  /** Monotonic per-sender sequence, so a receiver can spot gaps. */
  seq: number;
  /** Unix ms at the ORIGIN (the drone for capture messages). Propagated
   *  unchanged through every hop — this is what makes end-to-end latency
   *  measurable per segment. */
  originTs: number;
  from: ComponentId;
  payload: T;
}

// ---------------------------------------------------------------------------
// §2 Uplink — drone → gateway → proxy → core
// ---------------------------------------------------------------------------

export interface DroneHello {
  kind: 'drone-hello';
  droneId: number;
  station: DroneStation;
  /** Free-form model/firmware string for the operator panel. */
  model: string;
  mode: LinkMode;
}

/** One drone's live state. Drives the control tower's map and the board's camera. */
export interface DroneTelemetry {
  kind: 'telemetry';
  droneId: number;
  /** Carried on every fix so a viewer that joined late still knows the fleet
   *  layout without waiting for a hello it already missed. */
  station: DroneStation;
  gps: Gps;
  headingDeg: number;
  /** Ground speed, m/s. */
  speed: number;
  batteryPct: number;
  /** Unix ms at the drone. */
  t: number;
}

/**
 * A slice of captured video. The drone cuts its stream into fixed slices and
 * ships each one with the poses covering it — the pose is what lets the
 * reconstruction skip re-deriving camera positions from scratch.
 */
export interface VideoSegment {
  kind: 'video-segment';
  droneId: number;
  /** Per-drone increasing slice number. */
  seq: number;
  codec: 'h265';
  startedAt: number;
  durationMs: number;
  /** Where the bytes live. In demo mode this is a file under res/static/video. */
  uri: string;
  /**
   * The same capture in a rendition a browser can decode, or null when `uri`
   * already is one. The uplink artifact is HEVC because that is what the radio
   * carries, and most browsers cannot play HEVC in MP4 — so the operator video
   * panel needs a second address for the identical footage rather than a
   * silently black player.
   */
  previewUri: string | null;
  /**
   * The footage runs opposite to the leg being flown and must be played
   * backwards. Demo material: the centre pass was only ever flown outbound, so
   * the return leg reuses it in reverse rather than showing the drone flying
   * away while it comes home.
   */
  reverse: boolean;
  bytes: number;
  /** Poses sampled across the slice, oldest first. */
  poses: DroneTelemetry[];
}

export type UplinkMessage = DroneHello | DroneTelemetry | VideoSegment;

// ---------------------------------------------------------------------------
// §3 Downlink — core → drone (control) / core → clients (situation)
// ---------------------------------------------------------------------------

/** Control tower assigns a GPS route. Goes core → drone. */
export interface AssignRoute {
  kind: 'assign-route';
  droneId: number;
  waypoints: Gps[];
  /** Fly the route back and forth instead of stopping at the last waypoint. */
  loop: boolean;
}

/** Operator's live stick input, when they take over from the route. */
export interface ManualControl {
  kind: 'manual-control';
  droneId: number;
  /** -1..1 each. */
  forward: number;
  yaw: number;
  climb: number;
}

export type ControlMessage = AssignRoute | ManualControl;

/**
 * Where the mission stands. The control tower and the board both render this,
 * and the demo scenario is expressed entirely through these phases:
 * idle → assigned → awaiting-drone → active.
 */
export type MissionPhase = 'idle' | 'assigned' | 'awaiting-drone' | 'active';

export interface MissionStatus {
  kind: 'mission-status';
  phase: MissionPhase;
  /** Operator-facing line, already localized. */
  message: string;
  /** Drones currently connected to the core. */
  dronesOnline: number;
  /** Set while the phase is time-bound (awaiting-drone), else null. */
  etaSeconds: number | null;
}

// ---------------------------------------------------------------------------
// §4 Reconstruction stream — the delay pattern
// ---------------------------------------------------------------------------
//
// The core cuts the flight into SEGMENTS and drives each one up a ladder of
// LEVELS: a low training-step result is confirmed and pushed as soon as the
// drone has passed, then refined — while the NEXT segment starts its own first
// level. Scheduling lives in the core ONLY; clients render what arrives.

/** Where a chunk lands in the shared frame. */
export interface SplatAlign {
  /** GPS anchor to place the chunk at; null uses the scene origin. */
  anchor: Gps | null;
  position: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
}

export interface SplatChunk {
  kind: 'splat-chunk';
  id: string;
  /** Capture segment this reconstructs — a piece of the scene, not a copy. */
  segment: number;
  /** Refinement level; 1 lands first, a higher one REPLACES it. */
  level: number;
  /** Training steps behind this level. */
  steps: number;
  /** What the commander can make out at this level (report 표 8). */
  label: string;
  /** No further level will arrive for this segment. */
  final: boolean;
  url: string;
  bytes: number;
  align: SplatAlign;
}

/**
 * What the main drone camera is showing right now. Derived from the video the
 * drone is uplinking, so the control tower's MAIN CAM panel shows the actual
 * capture rather than a placeholder.
 */
export interface CameraFeed {
  kind: 'camera-feed';
  droneId: number;
  station: DroneStation;
  /** The uplinked artifact (HEVC on the wire). */
  uri: string;
  /** Browser-playable rendition of the same footage, when one exists. */
  previewUri: string | null;
  /** Play it backwards — see VideoSegment.reverse. */
  reverse: boolean;
  codec: 'h265';
  /** Unix ms at the drone when this slice started. */
  startedAt: number;
  durationMs: number;
}

/** Segment-level view of the same stream, for progress UI. */
export interface SegmentStatus {
  index: number;
  /** Highest level delivered; 0 = queued/processing. */
  level: number;
  levels: number;
  steps: number;
  label: string;
}

// ---------------------------------------------------------------------------
// §5 Detection
// ---------------------------------------------------------------------------

export interface DetectionResult {
  kind: 'detection';
  id: string;
  category: 'person' | 'danger';
  gps: Gps;
  confidence: number;
  label: string;
  /** Segment the detection was found in, so the board can gate it on arrival. */
  segment: number;
}

// ---------------------------------------------------------------------------
// §6 Link health — what the badges show
// ---------------------------------------------------------------------------

export interface LinkStatus {
  kind: 'link-status';
  /** Which hop this describes, e.g. 'drone→gateway'. */
  hop: string;
  connected: boolean;
  mode: LinkMode;
  /** Round-trip ms, null when never measured. */
  latencyMs: number | null;
  /** Uplink bitrate estimate, Mbps. */
  mbps: number | null;
}

export interface ServerStatus {
  kind: 'server-status';
  connected: boolean;
  receiving: boolean;
  chunks: number;
  detections: number;
  lastSeq: number;
  latencyMs: number | null;
  segments: SegmentStatus[];
}

// ---------------------------------------------------------------------------
// §7 Model API (REST) — core → skylens_model
// ---------------------------------------------------------------------------
//
// The only request/response surface in the system. Everything else is push.

/** POST /recon/jobs */
export interface ReconJobRequest {
  segment: number;
  /** Video slices making up this segment. */
  sources: Array<{ uri: string; poses: DroneTelemetry[] }>;
  /**
   * The stretch of route this segment covers, sampled start to end.
   *
   * A reconstruction built from images has shape but no SIZE — structure from
   * motion cannot recover metric scale — so whatever comes back has to be
   * scaled against something measured before it can be drawn on a map. This is
   * that measurement, and only the core has it: a segment is closed by arc
   * length along the assigned route, and the core is what does the closing.
   *
   * It is sent separately from `sources` because a segment closes when the
   * aircraft LEAVES it, which is before its video has finished uploading: the
   * first refinement level is dispatched with no slices at all, and it still
   * has to land in the right place at the right size.
   */
  track?: Gps[];
  /** Training steps to run for this level. */
  steps: number;
  /** Frame of the FIRST segment, forced onto every later one so segments line
   *  up in one space (see 중간보고서 Ⅲ-1-바). Null for the first job. */
  anchorFrame: string | null;
}

/** POST /detect/jobs */
export interface DetectJobRequest {
  segment: number;
  sources: Array<{ uri: string; poses: DroneTelemetry[] }>;
}

export interface JobAccepted {
  jobId: string;
  /** Server-side queue position at accept time. */
  queued: number;
}

export type JobState = 'queued' | 'running' | 'done' | 'failed';

/** GET /jobs/{id} */
export interface JobStatus {
  jobId: string;
  state: JobState;
  progress: number;
  /** Present when state === 'done'. */
  result: ReconJobResult | DetectJobResult | null;
  error: string | null;
}

export interface ReconJobResult {
  kind: 'recon-result';
  segment: number;
  steps: number;
  url: string;
  bytes: number;
  align: SplatAlign;
  /** Frame this job established; later jobs must be given it as anchorFrame. */
  anchorFrame: string;
}

export interface DetectJobResult {
  kind: 'detect-result';
  segment: number;
  detections: DetectionResult[];
}

// ---------------------------------------------------------------------------
// §8 Unions
// ---------------------------------------------------------------------------

/** Anything the core pushes to a viewer (control tower or situation board). */
export type ViewerMessage =
  // The route in force. It travels to viewers as well as to the drone: an
  // operator who cannot see the line they planned has no way to tell whether
  // the aircraft is following it.
  | AssignRoute
  | SplatChunk
  | DetectionResult
  | DroneTelemetry
  | CameraFeed
  | MissionStatus
  | ServerStatus
  | LinkStatus;

/**
 * Every kind a viewer may be handed — the single source for the hops that
 * filter by kind.
 *
 * Three components each kept their own copy of this list: the browser stream,
 * the client relay's upstream, and the relay's replay cache. Each one silently
 * dropped anything missing from it, so adding `assign-route` to the union left
 * the situation board without the planned track while every component reported
 * healthy. The `satisfies` below plus the exhaustiveness check underneath make
 * a forgotten kind a compile error instead.
 */
export const VIEWER_MESSAGE_KINDS = [
  'assign-route',
  'splat-chunk',
  'detection',
  'telemetry',
  'camera-feed',
  'mission-status',
  'server-status',
  'link-status',
] as const satisfies readonly ViewerMessage['kind'][];

/** Compile error when a ViewerMessage kind is missing from the list above. */
type MissingViewerKind = Exclude<ViewerMessage['kind'], (typeof VIEWER_MESSAGE_KINDS)[number]>;
const _viewerKindsAreComplete: MissingViewerKind extends never ? true : never = true;
void _viewerKindsAreComplete;

/** Is this a kind a viewer understands? */
export function isViewerMessageKind(kind: unknown): kind is ViewerMessage['kind'] {
  return (
    typeof kind === 'string' &&
    (VIEWER_MESSAGE_KINDS as readonly string[]).includes(kind)
  );
}

export const IDENTITY_ALIGN: SplatAlign = {
  anchor: null,
  position: [0, 0, 0],
  rotation: [0, 0, 0, 1],
  scale: [1, 1, 1],
};
