// Scripted drone, for exercising the gateway without a Tauri build.
//
// Sends the real uplink shapes from src/shared/protocol.ts: a DroneHello,
// then telemetry on a tick, then a VideoSegment that REFERENCES its bytes by uri
// (res/static/demo/...) rather than inlining them — the same discipline the real
// drone follows so a frame stays small.
//
//   npx tsx src/test/gateway/fakeDrone.ts                       # relay via gateway
//   npx tsx src/test/gateway/fakeDrone.ts ws://127.0.0.1:8082/direct
//
// Env: DRONE_ID, DRONE_TICK_MS, DRONE_SEGMENT_EVERY (ticks), DRONE_FRAMES (exit after N).

import process from 'node:process';
import { WebSocket } from 'ws';
import type {
  DroneHello,
  DroneTelemetry,
  Envelope,
  UplinkMessage,
  VideoSegment,
} from '../../shared/protocol.ts';

const url = process.argv[2] ?? 'ws://127.0.0.1:8081/uplink';
const droneId = Number(process.env.DRONE_ID ?? 1);
const tickMs = Number(process.env.DRONE_TICK_MS ?? 500);
const segmentEvery = Number(process.env.DRONE_SEGMENT_EVERY ?? 4);
const maxFrames = Number(process.env.DRONE_FRAMES ?? 0);

let seq = 0;
let ticks = 0;
let segment = 0;

function send(ws: WebSocket, payload: UplinkMessage): void {
  seq += 1;
  const env: Envelope<UplinkMessage> = {
    seq,
    originTs: Date.now(),
    from: 'drone',
    payload,
  };
  ws.send(JSON.stringify(env));
  console.log(`[drone ${droneId}] -> ${payload.kind} seq=${seq}`);
}

function telemetry(): DroneTelemetry {
  // A slow arc over 서울 서대문구 홍제동, the report's reference site.
  const t = ticks * 0.0002;
  return {
    kind: 'telemetry',
    droneId,
    gps: { lat: 37.5866 + t, lon: 126.9436 + t * 0.5, alt: 80 + Math.sin(ticks / 4) * 5 },
    headingDeg: (ticks * 7) % 360,
    speed: 6.5,
    batteryPct: Math.max(5, 100 - ticks * 0.4),
    t: Date.now(),
  };
}

const ws = new WebSocket(url);

ws.on('open', () => {
  console.log(`[drone ${droneId}] connected to ${url}`);
  const hello: DroneHello = {
    kind: 'drone-hello',
    droneId,
    model: 'SkyLens-Demo/H265',
    mode: url.includes('/direct') ? 'webrtc' : 'relay',
  };
  send(ws, hello);

  const timer = setInterval(() => {
    if (ws.readyState !== WebSocket.OPEN) return;
    ticks += 1;
    const poses: DroneTelemetry[] = [telemetry()];
    send(ws, poses[0]);

    if (ticks % segmentEvery === 0) {
      segment += 1;
      const vs: VideoSegment = {
        kind: 'video-segment',
        droneId,
        seq: segment,
        codec: 'h265',
        startedAt: Date.now() - tickMs * segmentEvery,
        durationMs: tickMs * segmentEvery,
        uri: `res/static/demo/drone${droneId}/seg${String(segment).padStart(4, '0')}.h265`,
        bytes: 1_450_000 + segment * 1000,
        poses,
      };
      send(ws, vs);
    }

    if (maxFrames > 0 && seq >= maxFrames) {
      clearInterval(timer);
      console.log(`[drone ${droneId}] sent ${seq} frame(s), closing`);
      ws.close();
    }
  }, tickMs);
});

ws.on('message', (data) => {
  try {
    const env = JSON.parse(data.toString()) as Envelope<{ kind?: string; hop?: string }>;
    const p = env.payload;
    if (p?.kind === 'link-status') console.log(`[drone ${droneId}] <- link-status ${p.hop}`);
    else console.log(`[drone ${droneId}] <- ${p?.kind ?? 'unknown'}`);
  } catch {
    console.log(`[drone ${droneId}] <- non-JSON`);
  }
});

ws.on('error', (err) => console.error(`[drone ${droneId}] error: ${err.message}`));
ws.on('close', () => {
  console.log(`[drone ${droneId}] closed`);
  process.exit(0);
});
