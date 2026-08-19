// Scripted proxy+drone against the core's ws /uplink.
//
// Stands in for 드론 → 게이트웨이 → 프록시 so the core can be driven without any
// of them: it speaks the same Envelope<UplinkMessage> the proxy forwards, adds a
// proxy hop stamp, and — crucially — MOVES. The delay pattern is triggered by
// movement, so a harness that does not fly proves nothing.
//
//   npx tsx src/test/core/fakeUplink.ts
//   UPLINK_URL=ws://127.0.0.1:8080/uplink SPEED=15 npx tsx src/test/core/fakeUplink.ts
//
// Env: UPLINK_URL · DRONE_ID · TICK_MS · SPEED (m/s) · SLICE_EVERY (ticks)
//      ROUTE_M (built-in route length when the core has assigned none) · RUN_MS

import process from 'node:process';
import { WebSocket } from 'ws';
import type {
  AssignRoute,
  ControlMessage,
  DroneHello,
  DroneTelemetry,
  Envelope,
  UplinkMessage,
  VideoSegment,
} from '../../shared/protocol.ts';
import type { Gps } from '../../shared/geo.ts';
import { enuToGps, gpsToEnu } from '../../shared/geo.ts';

const url = process.env.UPLINK_URL ?? 'ws://127.0.0.1:8080/uplink';
const droneId = Number(process.env.DRONE_ID ?? 1);
const tickMs = Number(process.env.TICK_MS ?? 250);
const speed = Number(process.env.SPEED ?? 12);
const sliceEvery = Number(process.env.SLICE_EVERY ?? 4);
const routeM = Number(process.env.ROUTE_M ?? 240);
const runMs = Number(process.env.RUN_MS ?? 0);

/** 서울 서대문구 홍제동 — the report's reference site. */
const HOME: Gps = { lat: 37.5866, lon: 126.9436, alt: 80 };

/** Straight patrol leg, so arc length is trivially checkable by hand. */
let waypoints: Gps[] = [HOME, enuToGps({ e: routeM, n: 0, u: 0 }, HOME)];
let loop = true;

let seq = 0;
let ticks = 0;
let sliceSeq = 0;
let travelled = 0;
let leg = 0;
let backwards = false;
const poseBuffer: DroneTelemetry[] = [];

const ws = new WebSocket(url);

function send(payload: UplinkMessage): void {
  seq += 1;
  const env: Envelope<UplinkMessage> & { path: Array<Record<string, unknown>> } = {
    seq,
    originTs: Date.now(),
    from: 'drone',
    payload,
    // The proxy stamps every frame it forwards; mimic it so the core sees the
    // shape it will see in the real chain.
    path: [{ at: 'proxy', rx: Date.now(), tx: Date.now(), via: url }],
  };
  ws.send(JSON.stringify(env));
}

/** Position at `travelled` meters along the (possibly reversed) polyline. */
function positionAt(distance: number): { gps: Gps; headingDeg: number } {
  const pts = waypoints.map((w) => gpsToEnu(w, waypoints[0]));
  const spans: number[] = [];
  let total = 0;
  for (let i = 0; i + 1 < pts.length; i += 1) {
    const d = Math.hypot(pts[i + 1].e - pts[i].e, pts[i + 1].n - pts[i].n);
    spans.push(d);
    total += d;
  }
  let along = distance;
  if (backwards) along = total - distance;
  along = Math.min(Math.max(along, 0), total);

  let acc = 0;
  for (let i = 0; i < spans.length; i += 1) {
    if (along <= acc + spans[i] || i === spans.length - 1) {
      const t = spans[i] === 0 ? 0 : (along - acc) / spans[i];
      const e = pts[i].e + (pts[i + 1].e - pts[i].e) * t;
      const n = pts[i].n + (pts[i + 1].n - pts[i].n) * t;
      const heading =
        (Math.atan2(pts[i + 1].e - pts[i].e, pts[i + 1].n - pts[i].n) * 180) / Math.PI;
      leg = i;
      return {
        gps: enuToGps({ e, n, u: 60 }, waypoints[0]),
        headingDeg: ((backwards ? heading + 180 : heading) + 360) % 360,
      };
    }
    acc += spans[i];
  }
  return { gps: waypoints[0], headingDeg: 0 };
}

function routeLength(): number {
  const pts = waypoints.map((w) => gpsToEnu(w, waypoints[0]));
  let total = 0;
  for (let i = 0; i + 1 < pts.length; i += 1) {
    total += Math.hypot(pts[i + 1].e - pts[i].e, pts[i + 1].n - pts[i].n);
  }
  return total;
}

function tick(): void {
  ticks += 1;
  const step = (speed * tickMs) / 1000;
  travelled += step;
  const total = routeLength();
  if (travelled >= total) {
    if (loop) {
      backwards = !backwards;
      travelled = 0;
      console.log(`[fake-uplink] turned around (leg ${leg}, ${backwards ? 'inbound' : 'outbound'})`);
    } else {
      travelled = total;
    }
  }

  const { gps, headingDeg } = positionAt(travelled);
  const t: DroneTelemetry = {
    kind: 'telemetry',
    droneId,
    gps,
    headingDeg,
    speed,
    batteryPct: Math.max(5, 100 - ticks * 0.05),
    t: Date.now(),
  };
  poseBuffer.push(t);
  send(t);

  if (ticks % sliceEvery === 0) {
    sliceSeq += 1;
    const slice: VideoSegment = {
      kind: 'video-segment',
      droneId,
      seq: sliceSeq,
      codec: 'h265',
      startedAt: Date.now() - tickMs * sliceEvery,
      durationMs: tickMs * sliceEvery,
      // Bytes are referenced, never inlined — the same discipline the real drone
      // follows so an uplink frame stays small.
      uri: `res/static/demo/videos/demo_center.mp4#${sliceSeq}`,
      bytes: 1_800_000,
      poses: poseBuffer.splice(0, poseBuffer.length),
    };
    send(slice);
    console.log(
      `[fake-uplink] slice #${sliceSeq} at ${travelled.toFixed(0)} m ` +
        `(${slice.poses.length} poses)`,
    );
  }
}

ws.on('open', () => {
  console.log(`[fake-uplink] connected to ${url}`);
  const hello: DroneHello = {
    kind: 'drone-hello',
    droneId,
    model: 'scripted harness (src/test/core/fakeUplink.ts)',
    mode: 'relay',
  };
  send(hello);
  console.log(
    `[fake-uplink] flying ${routeLength().toFixed(0)} m at ${speed} m/s, ` +
      `telemetry every ${tickMs} ms`,
  );
  const timer = setInterval(tick, tickMs);
  if (runMs > 0) {
    setTimeout(() => {
      clearInterval(timer);
      console.log('[fake-uplink] done');
      ws.close();
    }, runMs);
  }
});

ws.on('message', (data) => {
  let env: Partial<Envelope<ControlMessage>>;
  try {
    env = JSON.parse(data.toString()) as Partial<Envelope<ControlMessage>>;
  } catch {
    return;
  }
  const msg = (env.payload ?? env) as ControlMessage;
  if (msg.kind === 'assign-route') {
    const route = msg as AssignRoute;
    waypoints = route.waypoints;
    loop = route.loop;
    travelled = 0;
    backwards = false;
    console.log(
      `[fake-uplink] <- assign-route: ${route.waypoints.length} waypoints, ` +
        `${routeLength().toFixed(0)} m, loop=${route.loop}`,
    );
    return;
  }
  console.log(`[fake-uplink] <- ${msg.kind}`);
});

ws.on('close', () => {
  console.log('[fake-uplink] socket closed');
  process.exit(0);
});
ws.on('error', (err) => console.error(`[fake-uplink] ${err.message}`));
