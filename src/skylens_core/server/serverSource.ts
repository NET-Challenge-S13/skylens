// Server data source. In real operation the backend streams telemetry, human-
// detection results, and Gaussian-splat chunks; the client sends route commands.
// There is no live backend yet, so:
//   - demo mode  → a MOCK provider synthesizes the stream on a timer (so the full
//                  RECON flow plays out).
//   - real mode  → a stub connection point that idles "waiting for server" until
//                  a real endpoint is wired via connect(url).
//
// Both surfaces (SIM, RECON) create their own source; RECON consumes
// detections/splats, SIM sends assign-route and shows reception status.

import { CONFIG } from '../config.ts';
import { enuToGps } from '../geo.ts';
import { IDENTITY_ALIGN } from '../protocol.ts';
import type {
  AssignRoute,
  DetectionResult,
  DroneTelemetry,
  ServerStatus,
  SplatChunk,
} from '../protocol.ts';

export interface ServerSource {
  onStatus(cb: (s: ServerStatus) => void): void;
  onTelemetry(cb: (t: DroneTelemetry) => void): void;
  onDetection(cb: (d: DetectionResult) => void): void;
  onSplatChunk(cb: (c: SplatChunk) => void): void;
  /** Send a route command to the server (mock acks; real forwards). */
  assignRoute(cmd: AssignRoute): void;
  /** Begin producing/receiving. */
  start(): void;
  /** Wire a real backend endpoint (no-op in mock). */
  connect(url: string): void;
  readonly status: ServerStatus;
  dispose(): void;
}

export interface ServerSourceOptions {
  demo: boolean;
  /** Splat URL the mock advertises as the reconstructed scene. */
  splatUrl?: string | null;
}

type Cb<T> = (v: T) => void;

export function createServerSource(opts: ServerSourceOptions): ServerSource {
  const statusCbs: Cb<ServerStatus>[] = [];
  const teleCbs: Cb<DroneTelemetry>[] = [];
  const detCbs: Cb<DetectionResult>[] = [];
  const splatCbs: Cb<SplatChunk>[] = [];
  const timers: ReturnType<typeof setTimeout>[] = [];

  const status: ServerStatus = {
    kind: 'server-status',
    connected: false,
    receiving: false,
    chunks: 0,
    detections: 0,
    lastSeq: 0,
    latencyMs: null,
  };

  const emitStatus = (): void => {
    for (const cb of statusCbs) cb({ ...status });
  };

  const anchor = CONFIG.geo.anchor;
  // Detections sit within ~15 m of the anchor, so they land on the scene which is
  // auto-fit near the origin (anchor → scene origin).
  const MOCK_DETECTIONS: Array<Omit<DetectionResult, 'kind' | 'gps'> & { e: number; n: number; u: number }> = [
    { id: 'd1', category: 'person', confidence: 0.86, label: '생존자 추정 · 구역 A', e: -6, n: 4, u: 1 },
    { id: 'd2', category: 'person', confidence: 0.74, label: '생존자 추정 · 구역 B', e: 7, n: -3, u: 2 },
    { id: 'd3', category: 'danger', confidence: 0.91, label: '붕괴 위험구역 · 중앙', e: 0, n: 8, u: 3 },
  ];

  function runMock(): void {
    status.connected = true;
    status.receiving = true;
    status.latencyMs = 40;
    emitStatus();

    // The reconstructed scene arrives as one (or more) splat chunk(s).
    timers.push(
      setTimeout(() => {
        if (opts.splatUrl) {
          status.chunks += 1;
          status.lastSeq += 1;
          const chunk: SplatChunk = {
            kind: 'splat-chunk',
            id: 'chunk-0',
            url: opts.splatUrl,
            align: { ...IDENTITY_ALIGN },
          };
          for (const cb of splatCbs) cb(chunk);
          emitStatus();
        }
      }, 1200),
    );

    // Human-detection results trickle in as the drone scans.
    MOCK_DETECTIONS.forEach((d, i) => {
      timers.push(
        setTimeout(() => {
          status.detections += 1;
          status.lastSeq += 1;
          const det: DetectionResult = {
            kind: 'detection',
            id: d.id,
            category: d.category,
            confidence: d.confidence,
            label: d.label,
            gps: enuToGps({ e: d.e, n: d.n, u: d.u }, anchor),
          };
          for (const cb of detCbs) cb(det);
          emitStatus();
        }, 3000 + i * 2500),
      );
    });
  }

  return {
    onStatus: (cb) => statusCbs.push(cb),
    onTelemetry: (cb) => teleCbs.push(cb),
    onDetection: (cb) => detCbs.push(cb),
    onSplatChunk: (cb) => splatCbs.push(cb),
    assignRoute: () => {
      // Mock: accept silently. Real: forward to backend.
    },
    start: () => {
      if (opts.demo) runMock();
      else emitStatus(); // real: idle, "waiting for server"
    },
    connect: () => {
      // Real backend wiring goes here (WebSocket/HTTP). No-op for now.
    },
    get status() {
      return status;
    },
    dispose: () => {
      for (const t of timers) clearTimeout(t);
      timers.length = 0;
    },
  };
}
