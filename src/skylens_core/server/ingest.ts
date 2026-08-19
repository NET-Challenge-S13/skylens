// Uplink ingest — 프록시 → 코어 (ws /uplink).
//
// Two jobs, and nothing else:
//   1. remember what arrived (Store),
//   2. decide WHEN A SEGMENT CLOSES.
//
// (2) is the trigger of the delay pattern, and COMPONENTS.md §5.2 is explicit
// about it: **딜레이 패턴의 트리거는 시간이 아니라 드론의 이동량이다.** There is no
// timer in this file. A segment closes because the drone LEFT it — the arc length
// along the assigned route (or, routeless, the odometer) crossed a bucket edge.
// Everything downstream (orchestrator.ts) only ever hears "segment k closed".
//
// The socket is bidirectional: the proxy fans whatever the core writes back down
// to the drones, so control messages (assign-route, manual-control) leave here
// too. The proxy also pushes its own LinkStatus up this socket; that is not an
// UplinkMessage, so it is accepted separately and forwarded to viewers as-is.

import type { WebSocket, WebSocketServer } from 'ws';
import type {
  AssignRoute,
  ControlMessage,
  DroneTelemetry,
  Envelope,
  LinkStatus,
  UplinkMessage,
  VideoSegment,
} from '../../shared/protocol.ts';
import type { Gps } from '../../shared/geo.ts';
import type { DroneRecord, SegmentRecord, StampedEnvelope } from './types.ts';
import type { Store } from './store.ts';
import { groundDistanceM, segmentIndexFor } from './segmenter.ts';

export interface IngestEvents {
  onDroneUp: (drone: DroneRecord) => void;
  onDroneGone: (droneId: number) => void;
  onTelemetry: (t: DroneTelemetry) => void;
  /** The drone has LEFT this segment. The only delay-pattern trigger there is. */
  onSegmentClosed: (seg: SegmentRecord) => void;
  onLinkStatus: (s: LinkStatus) => void;
  /** The route in force, replayed to an uplink that attaches after it was
   *  assigned — 데모 시나리오 3~4 is exactly that order (경로 지정 → 드론 도착). */
  currentRoute: () => AssignRoute | null;
}

export interface IngestCounters {
  sockets: number;
  socketsSeen: number;
  frames: number;
  rejected: number;
  controlOut: number;
  /** Slices that arrived with no pose and were filed under the drone's segment. */
  poselessSlices: number;
}

interface UplinkSocket {
  ws: WebSocket;
  since: number;
  frames: number;
  /** Drones seen on this socket — dropped together when it closes. */
  drones: Set<number>;
}

export class Ingest {
  private store: Store;
  private segmentMeters: number;
  private events: IngestEvents;
  private sockets = new Set<UplinkSocket>();
  private seq = 0;
  private socketsSeen = 0;
  private frames = 0;
  private rejected = 0;
  private controlOut = 0;
  private poselessSlices = 0;

  constructor(opts: { store: Store; segmentMeters: number; events: IngestEvents }) {
    this.store = opts.store;
    this.segmentMeters = opts.segmentMeters;
    this.events = opts.events;
  }

  start(wss: WebSocketServer): void {
    wss.on('connection', (ws) => this.accept(ws));
  }

  counters(): IngestCounters {
    return {
      sockets: this.sockets.size,
      socketsSeen: this.socketsSeen,
      frames: this.frames,
      rejected: this.rejected,
      controlOut: this.controlOut,
      poselessSlices: this.poselessSlices,
    };
  }

  /** core → drone. The proxy relays this to whatever is attached to it. */
  sendControl(msg: ControlMessage, only?: WebSocket): number {
    this.seq += 1;
    const env: Envelope<ControlMessage> = {
      seq: this.seq,
      originTs: Date.now(),
      from: 'core',
      payload: msg,
    };
    const text = JSON.stringify(env);
    let sent = 0;
    for (const s of this.sockets) {
      if (only !== undefined && s.ws !== only) continue;
      if (s.ws.readyState !== 1) continue;
      try {
        s.ws.send(text);
        sent += 1;
      } catch (err) {
        console.warn(`[core] uplink send failed: ${String(err)}`);
      }
    }
    this.controlOut += sent;
    return sent;
  }

  /** A drone's arc bookkeeping restarts when the route it is measured against
   *  changes — arc length is meaningless across two different polylines. */
  resetArc(): void {
    for (const drone of this.store.drones.values()) {
      drone.currentSegment = null;
      drone.odometer = 0;
    }
  }

  stop(): void {
    for (const s of this.sockets) s.ws.close();
    this.sockets.clear();
  }

  // -------------------------------------------------------------------------

  private accept(ws: WebSocket): void {
    const conn: UplinkSocket = { ws, since: Date.now(), frames: 0, drones: new Set() };
    this.sockets.add(conn);
    this.socketsSeen += 1;
    console.log(`[core] uplink attached (${this.sockets.size} open)`);

    const route = this.events.currentRoute();
    if (route !== null) {
      this.sendControl(route, ws);
      console.log(`[core] replayed the assigned route to the new uplink`);
    }

    ws.on('message', (data) => {
      conn.frames += 1;
      this.receive(data.toString(), conn);
    });
    ws.on('close', () => {
      this.sockets.delete(conn);
      console.log(`[core] uplink detached after ${conn.frames} frame(s)`);
      for (const droneId of conn.drones) {
        this.store.dropDrone(droneId);
        this.events.onDroneGone(droneId);
      }
    });
    ws.on('error', (err) => console.warn(`[core] uplink socket error: ${err.message}`));
  }

  private receive(text: string, conn: UplinkSocket): void {
    let env: StampedEnvelope<UplinkMessage | LinkStatus>;
    try {
      env = JSON.parse(text) as StampedEnvelope<UplinkMessage | LinkStatus>;
    } catch {
      this.rejected += 1;
      console.warn('[core] dropped non-JSON frame on uplink');
      return;
    }
    const payload = env?.payload as (UplinkMessage | LinkStatus) | undefined;
    if (!payload || typeof payload.kind !== 'string') {
      this.rejected += 1;
      console.warn('[core] dropped uplink frame with no payload');
      return;
    }

    this.frames += 1;
    const c = this.store.counters;
    c.uplinkFrames += 1;
    if (typeof env.seq === 'number') c.lastSeq = env.seq;
    if (typeof env.originTs === 'number' && env.originTs > 0) {
      c.latencyMs = Math.max(0, Date.now() - env.originTs);
    }

    switch (payload.kind) {
      case 'drone-hello': {
        const rec = this.store.hello(payload);
        conn.drones.add(rec.droneId);
        console.log(`[core] drone ${rec.droneId} hello (${rec.model}, ${rec.mode})`);
        this.events.onDroneUp(rec);
        break;
      }
      case 'telemetry': {
        conn.drones.add(payload.droneId);
        this.telemetry(payload);
        break;
      }
      case 'video-segment': {
        conn.drones.add(payload.droneId);
        this.video(payload);
        break;
      }
      case 'link-status':
        this.events.onLinkStatus(payload);
        break;
      default: {
        // The tag came off the wire, so it can be anything; narrowing against
        // the union would leave `never` here.
        const kind: string = (payload as { kind: string }).kind;
        this.rejected += 1;
        console.warn(`[core] uplink frame with unknown kind "${kind}"`);
      }
    }
  }

  // -------------------------------------------------------------------------
  // Movement → segments
  // -------------------------------------------------------------------------

  private telemetry(msg: DroneTelemetry): void {
    const known = this.store.drones.get(msg.droneId);
    const previous = known?.last ?? null;
    const isNew = known === undefined;

    const rec = this.store.telemetry(msg);
    if (isNew) this.events.onDroneUp(rec);

    const arcM = this.advance(rec, previous, msg.gps);
    const index = segmentIndexFor(arcM, this.segmentMeters);

    if (rec.currentSegment === null) {
      rec.currentSegment = index;
      this.store.segment(index).passes += 1;
    } else if (index !== rec.currentSegment) {
      const left = this.store.segment(rec.currentSegment);
      rec.currentSegment = index;
      this.store.segment(index).passes += 1;
      this.close(left, arcM);
    }

    this.events.onTelemetry(msg);
  }

  /** Arc length along the route when one is assigned, plain odometer otherwise.
   *  The scheduler downstream cannot tell the two apart, on purpose. */
  private advance(rec: DroneRecord, previous: DroneTelemetry | null, gps: Gps): number {
    const projected = this.store.route.project(gps);
    if (projected !== null) return projected.arcM;
    if (previous !== null) rec.odometer += groundDistanceM(previous.gps, gps);
    return rec.odometer;
  }

  private close(seg: SegmentRecord, arcM: number): void {
    seg.generation += 1;
    seg.lastClosedAt = Date.now();
    if (seg.state === 'open') seg.state = 'queued';
    console.log(
      `[core] segment ${seg.index} closed at ${arcM.toFixed(1)} m ` +
        `(pass ${seg.passes}, gen ${seg.generation}, ${seg.sources.length} slice(s), ` +
        `level ${seg.deliveredLevel})`,
    );
    this.events.onSegmentClosed(seg);
  }

  /**
   * A slice is filed under the place it was SHOT, not the place the drone is now
   * — on a 왕복 route those differ by a whole segment at the far end. The middle
   * pose decides, so a slice straddling a boundary lands on the side it mostly
   * covers.
   */
  private video(msg: VideoSegment): void {
    const bytes = typeof msg.bytes === 'number' ? msg.bytes : 0;
    this.store.video(msg, bytes);

    const poses = Array.isArray(msg.poses) ? msg.poses : [];
    let index: number | null = null;
    if (poses.length > 0) {
      const mid = poses[Math.floor(poses.length / 2)];
      const projected = this.store.route.project(mid.gps);
      if (projected !== null) index = segmentIndexFor(projected.arcM, this.segmentMeters);
    }
    if (index === null) {
      this.poselessSlices += poses.length === 0 ? 1 : 0;
      index = this.store.drones.get(msg.droneId)?.currentSegment ?? 0;
    }

    const added = this.store.addSlice(index, {
      uri: msg.uri,
      poses,
      droneId: msg.droneId,
      seq: msg.seq,
      bytes,
      receivedAt: Date.now(),
    });
    if (added) {
      console.log(
        `[core] slice #${msg.seq} (${msg.uri}, ${poses.length} pose(s)) → segment ${index}`,
      );
    }
  }
}
