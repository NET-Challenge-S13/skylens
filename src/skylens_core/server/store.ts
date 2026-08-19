// The whole memory of the system.
//
// COMPONENTS.md §3.4 / §7: 이번 단계에서는 DB를 붙이지 않고 인메모리로 보관하며,
// 세션이 끝나면 데이터는 사라진다. That is a decision, not a shortcut — nothing
// here is allowed to reach for a disk or a database, so restarting the core is
// always a clean slate.

import type {
  CameraFeed,
  DetectionResult,
  DroneHello,
  DroneTelemetry,
  SegmentStatus,
  SplatChunk,
  VideoSegment,
} from '../../shared/protocol.ts';
import type { Gps } from '../../shared/geo.ts';
import type { DroneRecord, SegmentRecord, SourceSlice } from './types.ts';
import { RouteTracker, segmentSpan } from './segmenter.ts';

export interface StoreCounters {
  uplinkFrames: number;
  telemetry: number;
  videoSlices: number;
  videoBytes: number;
  chunksSent: number;
  detections: number;
  reconJobsDone: number;
  reconJobsDropped: number;
  reconJobsFailed: number;
  lastSeq: number;
  /** origin -> core, measured from Envelope.originTs. */
  latencyMs: number | null;
}

export class Store {
  readonly drones = new Map<number, DroneRecord>();
  readonly segments = new Map<number, SegmentRecord>();
  readonly detections: DetectionResult[] = [];
  readonly route = new RouteTracker();
  /** Waypoints exactly as the control tower sent them. */
  routeWaypoints: Gps[] = [];
  routeDroneId: number | null = null;
  routeLoop = false;
  /** Frame the FIRST recon job established; forced onto every later job. */
  anchorFrame: string | null = null;
  /** Newest main-camera slice, replayed to a viewer that joins mid-flight. */
  cameraFeed: CameraFeed | null = null;

  readonly counters: StoreCounters = {
    uplinkFrames: 0,
    telemetry: 0,
    videoSlices: 0,
    videoBytes: 0,
    chunksSent: 0,
    detections: 0,
    reconJobsDone: 0,
    reconJobsDropped: 0,
    reconJobsFailed: 0,
    lastSeq: 0,
    latencyMs: null,
  };

  private telemetryHistory: number;
  private segmentMeters: number;

  constructor(opts: { telemetryHistory: number; segmentMeters: number }) {
    this.telemetryHistory = opts.telemetryHistory;
    this.segmentMeters = opts.segmentMeters;
  }

  // -------------------------------------------------------------------------
  // Drones
  // -------------------------------------------------------------------------

  hello(msg: DroneHello, announced = true): DroneRecord {
    const now = Date.now();
    const existing = this.drones.get(msg.droneId);
    if (existing) {
      existing.model = msg.model;
      existing.mode = msg.mode;
      existing.lastSeenAt = now;
      // A real hello promotes a record that was adopted from telemetry.
      if (announced) existing.announced = true;
      return existing;
    }
    const rec: DroneRecord = {
      droneId: msg.droneId,
      model: msg.model,
      mode: msg.mode,
      connectedAt: now,
      lastSeenAt: now,
      last: null,
      history: [],
      odometer: 0,
      currentSegment: null,
      slices: 0,
      bytes: 0,
      announced,
    };
    this.drones.set(msg.droneId, rec);
    return rec;
  }

  dropDrone(droneId: number): void {
    this.drones.delete(droneId);
  }

  /** Records the fix and returns the drone, or null for an unknown drone that
   *  never said hello (we adopt it — a dev drone may skip the handshake). */
  telemetry(msg: DroneTelemetry): DroneRecord {
    let rec = this.drones.get(msg.droneId);
    if (!rec) {
      rec = this.hello(
        {
          kind: 'drone-hello',
          droneId: msg.droneId,
          model: 'unknown (adopted from telemetry)',
          mode: 'relay',
        },
        // Overheard, not announced — see DroneRecord.announced.
        false,
      );
    }
    rec.last = msg;
    rec.lastSeenAt = Date.now();
    rec.history.push(msg);
    if (rec.history.length > this.telemetryHistory) rec.history.shift();
    this.counters.telemetry += 1;
    return rec;
  }

  // -------------------------------------------------------------------------
  // Segments
  // -------------------------------------------------------------------------

  segment(index: number): SegmentRecord {
    const existing = this.segments.get(index);
    if (existing) return existing;
    const [spanStartM, spanEndM] = segmentSpan(index, this.segmentMeters);
    const rec: SegmentRecord = {
      index,
      spanStartM,
      spanEndM,
      sources: [],
      passes: 0,
      generation: 0,
      firstSeenAt: Date.now(),
      lastClosedAt: null,
      deliveredLevel: 0,
      deliveredSteps: 0,
      label: '대기',
      state: 'open',
      chunk: null,
      detectGeneration: -1,
    };
    this.segments.set(index, rec);
    return rec;
  }

  addSlice(segIndex: number, slice: SourceSlice): boolean {
    const seg = this.segment(segIndex);
    if (seg.sources.some((s) => s.uri === slice.uri)) return false;
    seg.sources.push(slice);
    return true;
  }

  video(msg: VideoSegment, bytes: number): void {
    const rec = this.drones.get(msg.droneId);
    if (rec) {
      rec.slices += 1;
      rec.bytes += bytes;
    }
    this.counters.videoSlices += 1;
    this.counters.videoBytes += bytes;
  }

  putChunk(chunk: SplatChunk): void {
    const seg = this.segment(chunk.segment);
    seg.chunk = chunk;
    seg.deliveredLevel = chunk.level;
    seg.deliveredSteps = chunk.steps;
    seg.label = chunk.label;
    seg.state = chunk.final ? 'final' : 'settled';
    this.counters.chunksSent += 1;
  }

  addDetection(det: DetectionResult): boolean {
    if (this.detections.some((d) => d.id === det.id)) return false;
    this.detections.push(det);
    this.counters.detections = this.detections.length;
    return true;
  }

  /** Latest chunk per segment, oldest segment first — the catch-up replay a
   *  viewer gets when it connects late. */
  chunks(): SplatChunk[] {
    return [...this.segments.values()]
      .filter((s): s is SegmentRecord & { chunk: SplatChunk } => s.chunk !== null)
      .sort((a, b) => a.index - b.index)
      .map((s) => s.chunk);
  }

  segmentStatus(levels: number): SegmentStatus[] {
    return [...this.segments.values()]
      .sort((a, b) => a.index - b.index)
      .map((s) => ({
        index: s.index,
        level: s.deliveredLevel,
        levels,
        steps: s.deliveredSteps,
        label: s.label,
      }));
  }

  setRoute(droneId: number, waypoints: Gps[], loop: boolean): void {
    this.routeDroneId = droneId;
    this.routeWaypoints = waypoints;
    this.routeLoop = loop;
    this.route.setRoute(waypoints);
  }
}
