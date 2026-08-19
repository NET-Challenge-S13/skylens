// skylens_core — KOREN 내부망. 시스템의 중심 (COMPONENTS.md §3.4).
//
//   ws  /uplink   프록시 → 코어 (드론 텔레메트리·영상 슬라이스), 역방향으로 제어
//   ws  /viewer   코어 → 뷰어 (관제탑·현황판), 역방향으로 제어 메시지
//   GET /health   상태 전체 (미션·구간·큐·모델 API·전송)
//   그 외          관제탑 화면 (web.ts)
//
// The four responsibilities of §3.4 are four objects, wired here and nowhere
// else:
//
//   Ingest        무엇이 도착했는가 + 구간이 언제 닫히는가 (이동량 기준)
//   Store         전부 인메모리. 프로세스가 죽으면 데이터도 죽는다 — 이번 단계의 결정
//   Orchestrator  딜레이 패턴. 스케줄 결정권은 오직 여기
//   Distributor   배포. WebSocket 뒤에 WebRTC 이음매를 감춘다 (distributor.ts)
//
// Run:  npx tsx src/skylens_core/server/index.ts
//       SKYLENS_CORE_WEB_MODE=off npx tsx src/skylens_core/server/index.ts

import http from 'node:http';
import process from 'node:process';
import express from 'express';
import { WebSocketServer } from 'ws';
import type {
  AssignRoute,
  ControlMessage,
  DetectionResult,
  LinkStatus,
  ServerStatus,
  SplatChunk,
} from '../../shared/protocol.ts';
import { loadConfig } from './config.ts';
import { WsDistributor, type ViewerSend } from './distributor.ts';
import { Ingest } from './ingest.ts';
import { buildLadder, topLevel } from './ladder.ts';
import { Mission } from './mission.ts';
import { ModelClient } from './modelClient.ts';
import { Orchestrator } from './orchestrator.ts';
import { Store } from './store.ts';
import { resolveSite, type Site } from './site.ts';
import { mountWeb } from './web.ts';

const cfg = loadConfig();
const startedAt = Date.now();
const ladder = buildLadder(cfg.levelSteps);
const levels = topLevel(ladder);

const store = new Store({
  telemetryHistory: cfg.telemetryHistory,
  segmentMeters: cfg.segmentMeters,
});

const model = new ModelClient({
  baseUrl: cfg.modelUrl,
  pollMs: cfg.modelPollMs,
  jobTimeoutMs: cfg.modelJobTimeoutMs,
});

const distributor = new WsDistributor();

const mission = new Mission({
  assignedHoldMs: cfg.assignedHoldMs,
  droneEtaSeconds: cfg.droneEtaSeconds,
  // Only drones that ANNOUNCED themselves count as on station; one still
  // flying to the site is in the store (so the tower can draw it) but has
  // not arrived. See DroneRecord.announced.
  dronesOnline: () => [...store.drones.values()].filter((d) => d.announced).length,
  onChange: (status) => distributor.broadcast(status),
});

const orchestrator = new Orchestrator({
  store,
  model,
  ladder,
  reconConcurrency: cfg.reconConcurrency,
  detectConcurrency: cfg.detectConcurrency,
  detect: cfg.detect,
  segmentMeters: cfg.segmentMeters,
  retryMs: cfg.modelRetryMs,
  maxAttempts: cfg.modelMaxAttempts,
  onChunk: (chunk: SplatChunk) => distributor.broadcast(chunk),
  onDetection: (det: DetectionResult) => distributor.broadcast(det),
});

const ingest = new Ingest({
  store,
  segmentMeters: cfg.segmentMeters,
  events: {
    onDroneUp: () => mission.droneConnected(),
    onDroneGone: () => mission.droneGone(),
    onTelemetry: (t) => distributor.broadcast(t),
    onCameraFeed: (f) => {
      store.cameraFeed = f;
      distributor.broadcast(f);
    },
    onSegmentClosed: (seg) => orchestrator.segmentClosed(seg),
    onLinkStatus: (s: LinkStatus) => distributor.broadcast(s),
    currentRoute: () =>
      store.routeWaypoints.length < 2 || store.routeDroneId === null
        ? null
        : {
            kind: 'assign-route',
            droneId: store.routeDroneId,
            waypoints: store.routeWaypoints,
            loop: store.routeLoop,
          },
  },
});

// ---------------------------------------------------------------------------
// Viewer surface
// ---------------------------------------------------------------------------

function serverStatus(): ServerStatus {
  const now = Date.now();
  let receiving = false;
  for (const d of store.drones.values()) {
    if (now - d.lastSeenAt < 3000) receiving = true;
  }
  return {
    kind: 'server-status',
    connected: true,
    receiving,
    chunks: store.counters.chunksSent,
    detections: store.detections.length,
    lastSeq: store.counters.lastSeq,
    latencyMs: store.counters.latencyMs,
    segments: store.segmentStatus(levels),
  };
}

/** A viewer that joins late must not see an empty world: replay everything the
 *  core has, in the order the board would have received it live. */
distributor.onJoin((send: ViewerSend) => {
  send(mission.status());
  for (const drone of store.drones.values()) if (drone.last) send(drone.last);
  if (store.routeWaypoints.length >= 2 && store.routeDroneId !== null) {
    send({
      kind: 'assign-route',
      droneId: store.routeDroneId,
      waypoints: store.routeWaypoints,
      loop: store.routeLoop,
    });
  }
  for (const chunk of store.chunks()) send(chunk);
  for (const det of store.detections) send(det);
  if (store.cameraFeed) send(store.cameraFeed);
  send(serverStatus());
});

distributor.onControl((msg: ControlMessage, send: ViewerSend) => {
  if (msg.kind === 'assign-route') {
    assignRoute(msg, send);
    return;
  }
  // manual-control: straight through to the drone, no mission transition — the
  // operator taking the sticks does not end the task.
  const sent = ingest.sendControl(msg);
  if (sent === 0) console.warn('[core] manual-control dropped: no uplink attached');
});

function assignRoute(msg: AssignRoute, send: ViewerSend): void {
  const waypoints = Array.isArray(msg.waypoints) ? msg.waypoints : [];
  if (waypoints.length < 2) {
    console.warn(`[core] assign-route ignored: ${waypoints.length} waypoint(s)`);
    send(mission.status());
    return;
  }
  // The UI may omit `loop`; the demo scenario is 지정 경로 왕복 반복, so a route
  // is a patrol unless the operator says otherwise.
  const loop = msg.loop !== false;
  const droneId = typeof msg.droneId === 'number' ? msg.droneId : 1;

  store.setRoute(droneId, waypoints, loop);
  // Segment indices are arc length along THIS route; the old arc means nothing.
  ingest.resetArc();
  console.log(
    `[core] route assigned to drone ${droneId}: ${waypoints.length} waypoints, ` +
      `${store.route.lengthM.toFixed(0)} m → ` +
      `${Math.ceil(store.route.lengthM / cfg.segmentMeters)} segment(s), loop=${loop}`,
  );

  // Viewers draw the line so the operator can see what the fleet was given.
  distributor.broadcast({ kind: 'assign-route', droneId, waypoints, loop });

  const forwarded = ingest.sendControl({ ...msg, loop, droneId });
  if (forwarded === 0) {
    console.log('[core] no uplink attached yet — the drone gets the route when it connects');
  }
  mission.routeAssigned();
  send(mission.status());
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

const app = express();

/**
 * Where this core sits. The control tower opens its route planner here, so it
 * lands on the operating area instead of a constant compiled into the client.
 * Resolved once at startup; see site.ts for the order of trust.
 */
let site: Site = { gps: cfg.siteFallback, source: 'fallback', label: null };
void resolveSite({
  configured: cfg.site,
  fallback: cfg.siteFallback,
  lookup: cfg.siteLookup,
  lookupUrl: cfg.siteLookupUrl,
  timeoutMs: cfg.siteLookupTimeoutMs,
}).then((resolved) => {
  site = resolved;
  console.log(
    `[core] site ${resolved.gps.lat.toFixed(4)}, ${resolved.gps.lon.toFixed(4)}` +
      ` (${resolved.source}${resolved.label ? ` · ${resolved.label}` : ''})`,
  );
});

app.get('/site', (_req, res) => {
  res.json(site);
});

app.get('/health', (_req, res) => {
  res.json({
    component: 'skylens_core',
    ok: true,
    uptimeSec: Math.round((Date.now() - startedAt) / 1000),
    listen: { port: cfg.port, uplink: cfg.uplinkPath, viewer: cfg.viewerPath },
    web: web.describe(),
    mission: mission.status(),
    delayPattern: {
      segmentMeters: cfg.segmentMeters,
      trigger: 'drone movement (arc length), never a clock',
      ladder: ladder.map((l) => ({ level: l.level, steps: l.steps, label: l.label })),
      reconConcurrency: cfg.reconConcurrency,
      detectConcurrency: cfg.detectConcurrency,
    },
    site,
    route: {
      droneId: store.routeDroneId,
      waypoints: store.routeWaypoints.length,
      lengthM: Math.round(store.route.lengthM),
      loop: store.routeLoop,
    },
    drones: [...store.drones.values()].map((d) => ({
      droneId: d.droneId,
      model: d.model,
      mode: d.mode,
      segment: d.currentSegment,
      odometerM: Math.round(d.odometer),
      slices: d.slices,
      lastSeenAgoMs: Date.now() - d.lastSeenAt,
    })),
    segments: store.segmentStatus(levels),
    jobs: orchestrator.counters(),
    uplink: ingest.counters(),
    distribution: distributor.counters(),
    model: model.health(),
    store: { ...store.counters, storage: 'in-memory (COMPONENTS.md §3.4 — 세션 종료 시 소멸)' },
  });
});

const web = mountWeb(app, cfg);

const server = http.createServer(app);
const uplinkWss = new WebSocketServer({ noServer: true });
const viewerWss = new WebSocketServer({ noServer: true });

server.on('upgrade', (req, socket, head) => {
  // Before anything else: a socket that resets mid-handshake must not raise an
  // unhandled 'error' and kill the core. Viewers come and go; the core stays.
  socket.on('error', (err) => {
    console.warn(`[core] upgrade socket error: ${err.message}`);
    socket.destroy();
  });
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  if (url.pathname === cfg.uplinkPath) {
    uplinkWss.handleUpgrade(req, socket, head, (ws) => uplinkWss.emit('connection', ws, req));
    return;
  }
  if (url.pathname === cfg.viewerPath) {
    viewerWss.handleUpgrade(req, socket, head, (ws) => viewerWss.emit('connection', ws, req));
    return;
  }
  if (web.upgrade(req, socket, head)) return;
  console.warn(`[core] rejected upgrade on ${url.pathname}`);
  socket.destroy();
});

ingest.start(uplinkWss);
distributor.start(viewerWss);
orchestrator.start();

// ServerStatus is a heartbeat; MissionStatus is pushed on change, plus on the
// tick while it is time-bound so a countdown actually counts down.
const statusTimer = setInterval(() => {
  distributor.broadcast(serverStatus());
  const status = mission.status();
  if (status.etaSeconds !== null) distributor.broadcast(status);
}, cfg.statusMs);

// Keeps /health honest about the model API even when no job is running.
const probeTimer = setInterval(() => void model.probe(), 5000);
void model.probe();

server.listen(cfg.port, cfg.host, () => {
  console.log(`[core] listening on http://${cfg.host}:${cfg.port}`);
  console.log(`[core]   uplink  ws://${cfg.host}:${cfg.port}${cfg.uplinkPath}`);
  console.log(`[core]   viewer  ws://${cfg.host}:${cfg.port}${cfg.viewerPath}`);
  console.log(`[core]   health  http://${cfg.host}:${cfg.port}/health`);
  console.log(`[core]   web     ${web.describe()}`);
  console.log(
    `[core] delay pattern: ${cfg.segmentMeters} m per segment, ladder ` +
      `${ladder.map((l) => l.steps).join(' → ')} steps, ${cfg.reconConcurrency} recon in flight`,
  );
  console.log(`[core] model API ${cfg.modelUrl}${cfg.demo ? ' (demo assets)' : ''}`);
});

function shutdown(signal: string): void {
  console.log(`[core] ${signal} — shutting down (in-memory state is discarded)`);
  clearInterval(statusTimer);
  clearInterval(probeTimer);
  orchestrator.stop();
  mission.stop();
  ingest.stop();
  distributor.stop();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1000).unref();
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
