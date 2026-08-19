// Scripted drone for webrtc mode: signal through the gateway, then send media
// STRAIGHT to the proxy. Proves the gateway drops out of the media path — after
// signal-ready this process never writes to the gateway socket again.
//
//   npx tsx src/test/gateway/fakeDroneWebrtc.ts
//
// Env: DRONE_ID, GATEWAY_SIGNAL_URL, DRONE_TICK_MS, DRONE_FRAMES

import process from 'node:process';
import { WebSocket } from 'ws';
import type {
  DroneHello,
  DroneTelemetry,
  Envelope,
  UplinkMessage,
  VideoSegment,
} from '../../shared/protocol.ts';
import type { SignalFrame } from '../../skylens_gateway/types.ts';

const signalUrl = process.env.GATEWAY_SIGNAL_URL ?? 'ws://127.0.0.1:8081/signal';
const droneId = Number(process.env.DRONE_ID ?? 7);
const tickMs = Number(process.env.DRONE_TICK_MS ?? 500);
const maxFrames = Number(process.env.DRONE_FRAMES ?? 8);

let signalSeq = 0;
let mediaSeq = 0;
let ticks = 0;

/** Anything arriving on the signalling socket: SignalFrame or a LinkStatus. */
interface AnyFrame {
  kind: string;
  sessionId?: string;
  direct?: string;
  hop?: string;
}

const signal = new WebSocket(signalUrl);

function sendSignal(frame: SignalFrame): void {
  signalSeq += 1;
  const env: Envelope<SignalFrame> = {
    seq: signalSeq,
    originTs: Date.now(),
    from: 'drone',
    payload: frame,
  };
  signal.send(JSON.stringify(env));
  console.log(`[drone ${droneId}] signal -> ${frame.kind}`);
}

signal.on('open', () => console.log(`[drone ${droneId}] signalling via ${signalUrl}`));

signal.on('message', (data) => {
  let env: Envelope<AnyFrame>;
  try {
    env = JSON.parse(data.toString()) as Envelope<AnyFrame>;
  } catch {
    return;
  }
  const frame = env.payload;
  if (!frame || typeof frame.kind !== 'string') return;
  if (frame.kind === 'link-status') {
    console.log(`[drone ${droneId}] signal <- link-status ${frame.hop}`);
    return;
  }
  console.log(
    `[drone ${droneId}] signal <- ${frame.kind}${frame.direct ? ` direct=${frame.direct}` : ''}`,
  );

  if (frame.kind === 'signal-hello' && frame.sessionId) {
    sendSignal({
      kind: 'signal-offer',
      sessionId: frame.sessionId,
      droneId,
      mode: 'webrtc',
      sdp: `v=0\r\no=skylens-drone ${Date.now()} 1 IN IP4 0.0.0.0\r\ns=skylens-uplink\r\n`,
    });
    return;
  }

  if (frame.kind === 'signal-ready' && frame.direct) {
    console.log(
      `[drone ${droneId}] punch done — media goes direct to ${frame.direct}; ` +
        `the gateway carries none of it`,
    );
    openMedia(frame.direct);
  }
});

signal.on('error', (err) => console.error(`[drone ${droneId}] signalling error: ${err.message}`));
signal.on('close', () => console.log(`[drone ${droneId}] signalling closed`));

function openMedia(url: string): void {
  const media = new WebSocket(url);

  const push = (payload: UplinkMessage): void => {
    mediaSeq += 1;
    const env: Envelope<UplinkMessage> = {
      seq: mediaSeq,
      originTs: Date.now(),
      from: 'drone',
      payload,
    };
    media.send(JSON.stringify(env));
    console.log(`[drone ${droneId}] direct -> ${payload.kind} seq=${mediaSeq}`);
  };

  media.on('open', () => {
    console.log(`[drone ${droneId}] direct media socket open: ${url}`);
    const hello: DroneHello = {
      kind: 'drone-hello',
      droneId,
      model: 'SkyLens-Demo/H265',
      mode: 'webrtc',
    };
    push(hello);

    const timer = setInterval(() => {
      if (media.readyState !== WebSocket.OPEN) return;
      ticks += 1;
      const pose: DroneTelemetry = {
        kind: 'telemetry',
        droneId,
        gps: { lat: 37.5866 + ticks * 0.0002, lon: 126.9436 + ticks * 0.0001, alt: 80 },
        headingDeg: (ticks * 7) % 360,
        speed: 6.5,
        batteryPct: 100 - ticks,
        t: Date.now(),
      };
      push(pose);
      if (ticks % 3 === 0) {
        const vs: VideoSegment = {
          kind: 'video-segment',
          droneId,
          seq: ticks / 3,
          codec: 'h265',
          startedAt: Date.now() - tickMs * 3,
          durationMs: tickMs * 3,
          uri: `res/static/demo/drone${droneId}/seg${String(ticks / 3).padStart(4, '0')}.h265`,
          bytes: 1_450_000,
          poses: [pose],
        };
        push(vs);
      }
      if (maxFrames > 0 && mediaSeq >= maxFrames) {
        clearInterval(timer);
        media.close();
        signal.close();
        setTimeout(() => process.exit(0), 200);
      }
    }, tickMs);
  });

  media.on('error', (err) => console.error(`[drone ${droneId}] direct error: ${err.message}`));
  media.on('close', () => console.log(`[drone ${droneId}] direct media socket closed`));
}
