/// <reference types="node" />
// A scripted stand-in for skylens_core, used to verify skylens_client end to end
// without waiting on the real orchestrator (and without a GPU).
//
// It speaks the REAL contract — shared/protocol.ts ViewerMessage on
// ws://localhost:8080/viewer — and streams the REAL demo assets in
// res/static/demo, cut into segment × level by split_segments.py. What it does
// NOT do is decide anything: the ladder, the stagger, and the labels all come
// out of segments.json, exactly as the core will read them from its own ladder.
//
// The stagger is the delay pattern: segment k's later levels deliberately land
// AFTER segment k+1's first level, so the board is proved to handle overlapping
// refinement rather than a tidy sequence.
//
// Run:  npx tsx src/test/client/fakeCore.ts
// Env:  FAKE_CORE_PORT (8080) · FAKE_MAX_LEVEL (all) · FAKE_SPEED (1.0)
//       FAKE_FIRST (1.2s) · FAKE_PERIOD (4s) · FAKE_LEVEL_DELAYS ("0,2,5,9")

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { WebSocketServer } from 'ws';
import type { WebSocket } from 'ws';
import type {
  DetectionResult,
  DroneTelemetry,
  Envelope,
  LinkStatus,
  MissionStatus,
  SegmentStatus,
  ServerStatus,
  SplatChunk,
  ViewerMessage,
} from '../../shared/protocol.ts';
import { IDENTITY_ALIGN } from '../../shared/protocol.ts';
import { enuToGps } from '../../shared/geo.ts';

interface Manifest {
  segments: Array<{
    index: number;
    levels: Array<{ level: number; steps: number; label: string; url: string }>;
  }>;
}

const PORT = Number(process.env.FAKE_CORE_PORT ?? 8080);
const SPEED = Number(process.env.FAKE_SPEED ?? 1);
const FIRST = Number(process.env.FAKE_FIRST ?? 1.2);
const PERIOD = Number(process.env.FAKE_PERIOD ?? 4);
const MAX_LEVEL = Number(process.env.FAKE_MAX_LEVEL ?? 99);
const LEVEL_DELAYS = (process.env.FAKE_LEVEL_DELAYS ?? '0,2,5,9')
  .split(',')
  .map((n) => Number(n.trim()));

// Same anchor the board converts GPS against (shared/viewer/config.ts).
const ANCHOR = { lat: 36.3685, lon: 127.3475, alt: 30 };
const DEMO_DIR = path.resolve('res/static/demo');
const ASSET_BASE = '/res/static/demo/';

const manifest = JSON.parse(
  fs.readFileSync(path.join(DEMO_DIR, 'segments.json'), 'utf8'),
) as Manifest;

const clients = new Set<WebSocket>();
const timers: NodeJS.Timeout[] = [];
let seq = 0;

// --- what the core would be holding ---------------------------------------
const segments: SegmentStatus[] = manifest.segments.map((s) => ({
  index: s.index,
  level: 0,
  // Ladder height is the CORE's: the board must never assume it.
  levels: s.levels.filter((l) => l.level <= MAX_LEVEL).length,
  steps: 0,
  label: '',
}));
const sentChunks: SplatChunk[] = [];
const sentDetections: DetectionResult[] = [];
let mission: MissionStatus = {
  kind: 'mission-status',
  phase: 'idle',
  message: '대기 중 · 경로 미지정',
  dronesOnline: 0,
  etaSeconds: null,
};
const links = new Map<string, LinkStatus>();
let telemetry: DroneTelemetry | null = null;

function serverStatus(): ServerStatus {
  return {
    kind: 'server-status',
    connected: true,
    receiving: sentChunks.length > 0,
    chunks: sentChunks.length,
    detections: sentDetections.length,
    lastSeq: seq,
    latencyMs: 38,
    segments: segments.map((s) => ({ ...s })),
  };
}

function send(ws: WebSocket, msg: ViewerMessage, wrap = false): void {
  const frame: ViewerMessage | Envelope<ViewerMessage> = wrap
    ? { seq: ++seq, originTs: Date.now(), from: 'core', payload: msg }
    : msg;
  try {
    ws.send(JSON.stringify(frame));
  } catch {
    /* client went away */
  }
}

function broadcast(msg: ViewerMessage, wrap = false): void {
  if (!wrap) seq += 1;
  for (const ws of clients) send(ws, msg, wrap);
}

/** The replay burst a joining viewer gets, in the order the core promises. */
function replay(ws: WebSocket): void {
  send(ws, mission);
  for (const l of links.values()) send(ws, l);
  if (telemetry) send(ws, telemetry, true);
  for (const c of sentChunks) send(ws, c);
  for (const d of sentDetections) send(ws, d);
  send(ws, serverStatus());
}

function at(seconds: number, fn: () => void): void {
  timers.push(setTimeout(fn, (seconds / SPEED) * 1000));
}

// --- the scripted mission --------------------------------------------------

function pushMission(next: MissionStatus): void {
  mission = next;
  broadcast(mission);
}

function pushLink(hop: string, connected: boolean, latencyMs: number | null, mbps: number | null): void {
  const l: LinkStatus = { kind: 'link-status', hop, connected, mode: 'relay', latencyMs, mbps };
  links.set(hop, l);
  broadcast(l);
}

function startFlight(): void {
  // 5 Hz telemetry on a slow circuit around the anchor.
  let t = 0;
  const tick = setInterval(() => {
    t += 0.2 * SPEED;
    const e = Math.cos(t * 0.25) * 18;
    const n = Math.sin(t * 0.25) * 18;
    telemetry = {
      kind: 'telemetry',
      droneId: 1,
      gps: enuToGps({ e, n, u: 22 }, ANCHOR),
      headingDeg: ((t * 0.25 * 180) / Math.PI + 90) % 360,
      speed: 4.5,
      batteryPct: Math.max(20, 96 - t * 0.4),
      t: Date.now(),
    };
    // Envelope-wrapped on purpose: the relay and the board both have to accept
    // wrapped and bare frames, and telemetry is the highest-rate one.
    broadcast(telemetry, true);
  }, 200);
  timers.push(tick);
}

function scheduleLadder(): void {
  manifest.segments.forEach((seg, si) => {
    const captured = FIRST + si * PERIOD;
    const levels = seg.levels.filter((l) => l.level <= MAX_LEVEL);
    levels.forEach((lv, li) => {
      const delay = LEVEL_DELAYS[Math.min(li, LEVEL_DELAYS.length - 1)] ?? 0;
      at(captured + delay, () => {
        const chunk: SplatChunk = {
          kind: 'splat-chunk',
          id: `seg${seg.index}-lv${lv.level}`,
          segment: seg.index,
          level: lv.level,
          steps: lv.steps,
          label: lv.label,
          final: li === levels.length - 1,
          url: ASSET_BASE + lv.url,
          bytes: 0,
          // Identity: this capture has no georeferenced placement, so the board
          // lands every segment on its own fit transform. All segments come from
          // ONE capture, so they share it.
          align: { ...IDENTITY_ALIGN },
        };
        // Newest level replaces the older one in the replay set, same rule the
        // real core and the relay cache follow.
        const prev = sentChunks.findIndex((c) => c.segment === chunk.segment);
        if (prev >= 0) sentChunks[prev] = chunk;
        else sentChunks.push(chunk);

        const st = segments[si];
        st.level = lv.level;
        st.steps = lv.steps;
        st.label = lv.label;

        console.log(`[fake-core] seg${seg.index} lv${lv.level} (${lv.steps} steps) → ${clients.size} viewer(s)`);
        broadcast(chunk);
        broadcast(serverStatus());
      });
    });
  });

  // Detections land after their segment's first level, so the board's
  // arrival-gating has something real to gate on.
  const DETS: Array<{ id: string; segment: number; category: 'person' | 'danger'; label: string; confidence: number; e: number; n: number; u: number }> = [
    { id: 'p1', segment: 0, category: 'person', label: '생존자 추정 · 구역 A', confidence: 0.86, e: -6, n: 4, u: 1 },
    { id: 'd1', segment: 1, category: 'danger', label: '붕괴 위험구역 · 중앙', confidence: 0.91, e: 0, n: 8, u: 3 },
    { id: 'p2', segment: 2, category: 'person', label: '생존자 추정 · 구역 B', confidence: 0.74, e: 7, n: -3, u: 2 },
  ];
  for (const d of DETS) {
    at(FIRST + d.segment * PERIOD + 1.0, () => {
      const det: DetectionResult = {
        kind: 'detection',
        id: d.id,
        category: d.category,
        gps: enuToGps({ e: d.e, n: d.n, u: d.u }, ANCHOR),
        confidence: d.confidence,
        label: d.label,
        segment: d.segment,
      };
      sentDetections.push(det);
      console.log(`[fake-core] detection ${det.id} (segment ${det.segment})`);
      broadcast(det);
      broadcast(serverStatus());
    });
  }
}

function main(): void {
  const server = http.createServer((req, res) => {
    if (req.url?.startsWith('/health')) {
      res.setHeader('content-type', 'application/json');
      res.end(
        JSON.stringify({
          component: 'fake-core',
          viewers: clients.size,
          chunks: sentChunks.length,
          detections: sentDetections.length,
          phase: mission.phase,
        }),
      );
      return;
    }
    res.statusCode = 404;
    res.end('fake core: only /viewer (ws) and /health');
  });

  const wss = new WebSocketServer({ server, path: '/viewer' });
  wss.on('connection', (ws) => {
    clients.add(ws);
    console.log(`[fake-core] viewer connected (${clients.size})`);
    replay(ws);
    ws.on('close', () => {
      clients.delete(ws);
      console.log(`[fake-core] viewer gone (${clients.size})`);
    });
    ws.on('error', () => clients.delete(ws));
  });

  server.listen(PORT, () => {
    console.log(`[fake-core] ws://localhost:${PORT}/viewer  (speed ${SPEED}x, maxLevel ${MAX_LEVEL})`);

    pushLink('drone→gateway', true, 24, 42);
    pushLink('gateway→proxy', true, 8, 41);
    pushLink('proxy→core', true, 3, 40);

    pushMission({
      kind: 'mission-status',
      phase: 'assigned',
      message: '태스크 지정 완료 · 드론 연결 대기',
      dronesOnline: 0,
      etaSeconds: 3,
    });
    at(0.5, () =>
      pushMission({
        kind: 'mission-status',
        phase: 'awaiting-drone',
        message: '드론이 현장에 도착해 연결되면 작업이 시작됩니다',
        dronesOnline: 0,
        etaSeconds: 2,
      }),
    );
    at(1.0, () => {
      pushMission({
        kind: 'mission-status',
        phase: 'active',
        message: '드론 연결됨 · 구간 복원 진행 중',
        dronesOnline: 1,
        etaSeconds: null,
      });
      startFlight();
      scheduleLadder();
    });
    broadcast(serverStatus());
  });

  const shutdown = (): void => {
    for (const t of timers) clearTimeout(t as unknown as NodeJS.Timeout);
    for (const ws of clients) ws.close();
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 500).unref();
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main();
