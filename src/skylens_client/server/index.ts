/// <reference types="node" />
// skylens_client — 현황판 클라이언트 웹서버 + WebRTC 연결 중계 (COMPONENTS.md §3.6).
//
// Port 8090. One process, four surfaces on it:
//
//   GET  /health              upstream state · connected boards · relayed counters
//   ws   /stream              board feed — every ViewerMessage the core pushes
//   any  /peerjs/**           PeerJS signalling (browser-facing WebRTC relay)
//   *                         the board web app (dev: proxy 5173 · prod: dist/)
//
// The component sits in the KOREN 외부망 and holds exactly ONE socket into the
// interior (ws://localhost:8080/viewer). Everything a browser needs is on this
// port, so a board is a single URL with no configuration:
//
//   http://<host>:8090/res/static/status.html
//
// Run:  npx tsx src/skylens_client/server/index.ts
//       SKYLENS_CLIENT_MODE=prod npx tsx src/skylens_client/server/index.ts

import http from 'node:http';
import express from 'express';
import { WebSocketServer } from 'ws';
import { loadConfig } from './config.ts';
import { createUpstream } from './upstream.ts';
import { createBoardHub } from './boards.ts';
import { mountPeerRelay } from './peerRelay.ts';
import { mountWebApp } from './webApp.ts';


function main(): void {
  const cfg = loadConfig();
  const startedAt = Date.now();

  const app = express();
  const server = http.createServer(app);

  const hub = createBoardHub(cfg.peerPath);
  const upstream = createUpstream(
    cfg.coreUrl,
    { minMs: cfg.backoffMinMs, maxMs: cfg.backoffMaxMs },
    {
      onMessage: (msg) => hub.broadcast(msg),
      onState: (state, detail) => {
        hub.setUpstream(state, detail, upstream.retries);
        console.log(`[client] upstream ${state} — ${detail}`);
      },
    },
  );

  // --- /health -------------------------------------------------------------
  // Registered before the catch-all web app so the proxy never swallows it.
  app.get('/health', (_req, res) => {
    const boards = hub.counters();
    res.json({
      component: 'skylens_client',
      ok: true,
      port: cfg.port,
      uptimeMs: Date.now() - startedAt,
      web: webApp.describe(),
      upstream: {
        url: cfg.coreUrl,
        state: upstream.state,
        detail: upstream.detail,
        since: upstream.since,
        retries: upstream.retries,
        received: upstream.received,
        malformed: upstream.malformed,
      },
      boards: {
        connected: boards.connected,
        seen: boards.seen,
        rooms: boards.rooms,
        streamPath: cfg.streamPath,
      },
      relayed: {
        total: boards.relayed,
        writes: boards.writes,
        replayed: boards.replayed,
        dropped: boards.dropped,
        byKind: boards.byKind,
        cached: boards.cached,
      },
      peer: peerRelay.counters(),
    });
  });

  // --- WebRTC signalling ---------------------------------------------------
  const peerRelay = mountPeerRelay(app, server, cfg.peerPath);

  // --- board web app (catch-all, so it goes last) --------------------------
  const webApp = mountWebApp(app, cfg);

  // --- board stream --------------------------------------------------------
  // noServer: this process routes upgrades itself, because three consumers
  // (board stream, PeerJS broker, Vite HMR) share one port.
  const streamWss = new WebSocketServer({ noServer: true });
  streamWss.on('connection', (ws, req) => {
    const url = new URL(req.url ?? '/', 'http://relay');
    const room = url.searchParams.get('room')?.trim() || 'default';
    hub.attach(ws, room);
  });

  server.on('upgrade', (req, socket, head) => {
    const pathname = new URL(req.url ?? '/', 'http://relay').pathname;
    if (pathname === cfg.streamPath) {
      streamWss.handleUpgrade(req, socket, head, (ws) => streamWss.emit('connection', ws, req));
      return;
    }
    if (pathname === peerRelay.wsPath || pathname.startsWith(`${cfg.peerPath}/`)) {
      peerRelay.handleUpgrade(req, socket, head);
      return;
    }
    webApp.upgrade(req, socket, head);
  });

  server.on('error', (err: Error & { code?: string }) => {
    if (err.code === 'EADDRINUSE') {
      console.error(
        `[client] 포트 ${cfg.port} 가 이미 사용 중입니다 — 기존 프로세스를 종료하거나 SKYLENS_CLIENT_PORT 로 다른 포트를 지정하세요.`,
      );
      process.exit(1);
    }
    console.error('[client] server error', err.message);
  });

  server.listen(cfg.port, cfg.host, () => {
    console.log(`[client] 현황판 서버 http://localhost:${cfg.port} (${cfg.mode})`);
    console.log(`[client]   board    http://localhost:${cfg.port}/res/static/status.html`);
    console.log(`[client]   stream   ws://localhost:${cfg.port}${cfg.streamPath}`);
    console.log(`[client]   peerjs   ws://localhost:${cfg.port}${peerRelay.wsPath}`);
    console.log(`[client]   upstream ${cfg.coreUrl}`);
    upstream.start();
  });

  const shutdown = (): void => {
    console.log('[client] shutting down');
    upstream.stop();
    hub.close();
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 1500).unref();
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main();
