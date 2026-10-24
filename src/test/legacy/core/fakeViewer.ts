// Scripted viewer against the core's ws /viewer.
//
// Stands in for both faces of that socket: a situation board (prints every
// ViewerMessage the core pushes) and a control tower (can send assign-route).
// Telemetry and server-status are summarised rather than printed per frame —
// otherwise the delay-pattern transcript drowns.
//
//   npx tsx src/test/core/fakeViewer.ts
//   npx tsx src/test/core/fakeViewer.ts --assign          # assign a route on connect
//   ASSIGN_AFTER_MS=3000 npx tsx src/test/core/fakeViewer.ts --assign
//
// Env: VIEWER_URL · ROUTE_M · ASSIGN_AFTER_MS · DRONE_ID · VERBOSE (1 = every frame)

import process from 'node:process';
import { WebSocket } from 'ws';
import type { AssignRoute, Envelope, ViewerMessage } from '../../shared/protocol.ts';
import type { Gps } from '../../shared/geo.ts';
import { enuToGps } from '../../shared/geo.ts';

const url = process.env.VIEWER_URL ?? 'ws://127.0.0.1:8080/viewer';
const assign = process.argv.includes('--assign');
const assignAfterMs = Number(process.env.ASSIGN_AFTER_MS ?? 500);
const routeM = Number(process.env.ROUTE_M ?? 240);
const droneId = Number(process.env.DRONE_ID ?? 1);
const verbose = process.env.VERBOSE === '1';

const HOME: Gps = { lat: 37.5866, lon: 126.9436, alt: 80 };
const started = Date.now();
let telemetry = 0;
let status = 0;

function stamp(): string {
  return `t+${((Date.now() - started) / 1000).toFixed(1)}s`;
}

const ws = new WebSocket(url);

ws.on('open', () => {
  console.log(`[viewer] connected to ${url}`);
  if (!assign) return;
  setTimeout(() => {
    const route: AssignRoute = {
      kind: 'assign-route',
      droneId,
      // A three-point patrol: out, dogleg, back — enough legs that arc length is
      // not just a straight line.
      waypoints: [
        HOME,
        enuToGps({ e: routeM * 0.5, n: 0, u: 0 }, HOME),
        enuToGps({ e: routeM, n: 0, u: 0 }, HOME),
      ],
      loop: true,
    };
    ws.send(JSON.stringify(route));
    console.log(`[viewer] ${stamp()} -> assign-route (${route.waypoints.length} waypoints)`);
  }, assignAfterMs);
});

ws.on('message', (data) => {
  let env: Partial<Envelope<ViewerMessage>> & Partial<ViewerMessage>;
  try {
    env = JSON.parse(data.toString()) as typeof env;
  } catch {
    console.warn('[viewer] non-JSON frame');
    return;
  }
  const msg = (typeof env.kind === 'string' ? env : env.payload) as ViewerMessage | undefined;
  if (!msg) return;

  switch (msg.kind) {
    case 'mission-status':
      console.log(
        `[viewer] ${stamp()} MISSION ${msg.phase.padEnd(14)} "${msg.message}" ` +
          `drones=${msg.dronesOnline} eta=${msg.etaSeconds ?? '-'}`,
      );
      break;
    case 'splat-chunk':
      console.log(
        `[viewer] ${stamp()} SPLAT   segment ${msg.segment} level ${msg.level} ` +
          `(${msg.steps} steps, ${(msg.bytes / 1024).toFixed(0)} KiB, "${msg.label}")` +
          `${msg.final ? ' FINAL' : ''} ${msg.url}`,
      );
      break;
    case 'detection':
      console.log(
        `[viewer] ${stamp()} DETECT  segment ${msg.segment} ${msg.category} ` +
          `${(msg.confidence * 100).toFixed(0)}% "${msg.label}" ` +
          `@ ${msg.gps.lat.toFixed(5)},${msg.gps.lon.toFixed(5)}`,
      );
      break;
    case 'link-status':
      console.log(`[viewer] ${stamp()} LINK    ${msg.hop} connected=${msg.connected}`);
      break;
    case 'server-status':
      status += 1;
      if (verbose || status % 4 === 0) {
        const ladder = msg.segments
          .map((s) => `s${s.index}:L${s.level}/${s.levels}`)
          .join(' ');
        console.log(
          `[viewer] ${stamp()} STATUS  receiving=${msg.receiving} chunks=${msg.chunks} ` +
            `detections=${msg.detections} lastSeq=${msg.lastSeq} ` +
            `latency=${msg.latencyMs ?? '-'}ms | ${ladder || 'no segments yet'}`,
        );
      }
      break;
    case 'telemetry':
      telemetry += 1;
      if (verbose) {
        console.log(
          `[viewer] ${stamp()} TELEM   drone ${msg.droneId} ` +
            `${msg.gps.lat.toFixed(5)},${msg.gps.lon.toFixed(5)} ${msg.speed} m/s`,
        );
      }
      break;
  }
});

ws.on('close', () => {
  console.log(`[viewer] closed after ${telemetry} telemetry frame(s)`);
  process.exit(0);
});
ws.on('error', (err) => console.error(`[viewer] ${err.message}`));
