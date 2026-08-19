// skylens_gateway — KOREN 외부망 진입점.
//
// Two modes, one process (COMPONENTS.md §3.2):
//   relay  (기본) — drone frames are forwarded to the proxy. See relay.ts.
//   webrtc        — hole-punch signalling only, media goes drone ↔ proxy. See signalling.ts.
//
// Run:  npx tsx src/skylens_gateway/index.ts
//       SKYLENS_GATEWAY_MODE=webrtc npx tsx src/skylens_gateway/index.ts

import http from 'node:http';
import process from 'node:process';
import express from 'express';
import { WebSocketServer } from 'ws';
import { dronePath, loadConfig } from './config.ts';
import { RelayGateway } from './relay.ts';
import { SignallingGateway } from './signalling.ts';

const cfg = loadConfig();
const startedAt = Date.now();

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ noServer: true });
const path = dronePath(cfg.mode);

const relay = cfg.mode === 'relay' ? new RelayGateway(cfg) : null;
const signalling = cfg.mode === 'webrtc' ? new SignallingGateway(cfg) : null;

app.get('/health', (_req, res) => {
  const upstream = relay ? relay.upstreamCounters() : signalling!.upstreamCounters();
  res.json({
    component: 'skylens_gateway',
    mode: cfg.mode,
    ok: true,
    uptimeSec: Math.round((Date.now() - startedAt) / 1000),
    listen: { port: cfg.port, dronePath: path },
    upstream: {
      // In webrtc mode this is the signalling control socket, not a media path.
      role: cfg.mode === 'relay' ? 'media-relay' : 'signalling-control',
      ...upstream,
    },
    counters: relay ? relay.counters() : signalling!.counters(),
  });
});

server.on('upgrade', (req, socket, head) => {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  if (url.pathname !== path) {
    console.warn(`[gateway] rejected upgrade on ${url.pathname} (mode ${cfg.mode} serves ${path})`);
    socket.destroy();
    return;
  }
  wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req));
});

if (relay) relay.start(wss);
if (signalling) signalling.start(wss);

server.listen(cfg.port, cfg.host, () => {
  console.log(`[gateway] mode=${cfg.mode} listening on http://${cfg.host}:${cfg.port}`);
  console.log(`[gateway] drones connect to ws://${cfg.host}:${cfg.port}${path}`);
  console.log(
    cfg.mode === 'relay'
      ? `[gateway] relaying to ${cfg.proxyIngressUrl}`
      : `[gateway] brokering hole punching with ${cfg.proxySignalUrl} (no media)`,
  );
  console.log(`[gateway] health: http://${cfg.host}:${cfg.port}/health`);
});

function shutdown(signal: string): void {
  console.log(`[gateway] ${signal} — shutting down`);
  relay?.stop();
  signalling?.stop();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1000).unref();
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
