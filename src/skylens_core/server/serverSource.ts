// Server data source. In real operation the backend streams telemetry, human-
// detection results, and Gaussian-splat chunks; the client sends route commands.
// There is no live backend yet, so:
//   - demo mode  → a MOCK provider synthesizes the stream on a timer (so the full
//                  STATUS flow plays out).
//   - real mode  → a stub connection point that idles "waiting for server" until
//                  a real endpoint is wired via connect(url).
//
// Both surfaces (CONTROL, STATUS) create their own source; STATUS consumes
// detections/splats, CONTROL sends assign-route and shows reception status.

import { CONFIG } from '../config.ts';
import { enuToGps } from '../geo.ts';
import { IDENTITY_ALIGN } from '../protocol.ts';
import type {
  AssignRoute,
  DetectionResult,
  DroneTelemetry,
  SegmentStatus,
  ServerStatus,
  SplatChunk,
} from '../protocol.ts';

/** segments.json, written by models/skylens/split_segments.py. */
interface SegmentManifest {
  segments: Array<{
    index: number;
    levels: Array<{ level: number; steps: number; label: string; url: string }>;
  }>;
}

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
  /** Segment × level manifest. When it loads, the mock streams the DELAY
   *  PATTERN (segments refined in overlapping stages) instead of one scene. */
  manifestUrl?: string | null;
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
    segments: [],
  };

  const emitStatus = (): void => {
    const snap: ServerStatus = { ...status, segments: status.segments.map((s) => ({ ...s })) };
    for (const cb of statusCbs) cb(snap);
  };

  const emitChunk = (chunk: SplatChunk, seg: SegmentStatus | null): void => {
    status.chunks += 1;
    status.lastSeq += 1;
    if (seg) {
      seg.level = chunk.level;
      seg.steps = chunk.steps;
      seg.label = chunk.label;
    }
    for (const cb of splatCbs) cb(chunk);
    emitStatus();
  };

  const anchor = CONFIG.geo.anchor;
  // Detections sit within ~15 m of the anchor, so they land on the scene which is
  // auto-fit near the origin (anchor → scene origin).
  const MOCK_DETECTIONS: Array<Omit<DetectionResult, 'kind' | 'gps'> & { e: number; n: number; u: number }> = [
    { id: 'd1', category: 'person', confidence: 0.86, label: '생존자 추정 · 구역 A', e: -6, n: 4, u: 1 },
    { id: 'd2', category: 'person', confidence: 0.74, label: '생존자 추정 · 구역 B', e: 7, n: -3, u: 2 },
    { id: 'd3', category: 'danger', confidence: 0.91, label: '붕괴 위험구역 · 중앙', e: 0, n: 8, u: 3 },
  ];

  /**
   * Delay-pattern stream. Segment k is "captured" at
   * `firstSegmentDelay + k * segmentPeriod`, then its levels land at that time
   * plus `levelDelays[i]`. Because the later delays exceed segmentPeriod, a
   * segment is still being refined while the next one delivers its first level
   * — the stagger the report describes.
   *
   * Returns false when the manifest isn't there (no local capture generated),
   * so the caller can fall back to the single-scene stream.
   */
  async function runDelayPattern(manifestUrl: string): Promise<boolean> {
    let manifest: SegmentManifest;
    try {
      const res = await fetch(manifestUrl);
      if (!res.ok) return false;
      manifest = (await res.json()) as SegmentManifest;
    } catch {
      return false;
    }
    if (!manifest.segments?.length) return false;

    const base = new URL(manifestUrl, window.location.href);
    const { firstSegmentDelay, segmentPeriod, levelDelays } = CONFIG.delayPattern;

    status.segments = manifest.segments.map((s) => ({
      index: s.index,
      level: 0,
      levels: s.levels.length,
      steps: 0,
      label: '',
    }));
    emitStatus();

    manifest.segments.forEach((seg, si) => {
      const captured = firstSegmentDelay + si * segmentPeriod;
      const track = status.segments[si];
      seg.levels.forEach((lv, li) => {
        const delay = levelDelays[Math.min(li, levelDelays.length - 1)] ?? 0;
        timers.push(
          setTimeout(
            () => {
              emitChunk(
                {
                  kind: 'splat-chunk',
                  id: `seg${seg.index}-lv${lv.level}`,
                  url: new URL(lv.url, base).toString(),
                  // Identity: the server hasn't computed a placement, so the
                  // client lands the chunk on its own fit transform. Every
                  // segment is cut from ONE capture, so they share it.
                  align: { ...IDENTITY_ALIGN },
                  segment: seg.index,
                  level: lv.level,
                  steps: lv.steps,
                  label: lv.label,
                  final: li === seg.levels.length - 1,
                },
                track,
              );
            },
            (captured + delay) * 1000,
          ),
        );
      });
    });
    return true;
  }

  /** Fallback: the whole reconstruction as a single chunk (CDN sample asset). */
  function runSingleScene(): void {
    timers.push(
      setTimeout(() => {
        if (!opts.splatUrl) return;
        emitChunk(
          {
            kind: 'splat-chunk',
            id: 'chunk-0',
            url: opts.splatUrl,
            align: { ...IDENTITY_ALIGN },
            segment: 0,
            level: 1,
            steps: 0,
            label: '',
            final: true,
          },
          null,
        );
      }, 1200),
    );
  }

  /** Human-detection results trickle in as the drone scans. */
  function runDetections(): void {
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

  function runMock(): void {
    status.connected = true;
    status.receiving = true;
    status.latencyMs = 40;
    emitStatus();

    void (async () => {
      const streamed = opts.manifestUrl ? await runDelayPattern(opts.manifestUrl) : false;
      if (!streamed) runSingleScene();
    })();

    runDetections();
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
