// End-to-end check of the demo drone against the scripted fake gateway.
//
//   npx tsx src/test/drone/scenario.ts [--seconds=55] [--mode=webrtc]
//
// Prints the transcript AS THE GATEWAY SEES IT (uplink frames off the wire),
// interleaved with the drone's own log, so the sequence
//   hello -> assign-route -> ~10 s transit -> telemetry -> video segments
// can be read off directly. Telemetry is sampled in the printout (it really
// flows at telemetryHz) so the transcript stays legible.
//
// Not a Playwright spec on purpose — it drives Node timers for the better part
// of a minute. `npm test` must not pick it up, hence the filename.

import process from 'node:process';
import type { Envelope, LinkMode, UplinkMessage } from '../../shared/protocol.ts';
import { DEFAULT_CONFIG, gatewayUrlFor } from '../../skylens_drone/core/config.ts';
import { DemoCapture } from '../../skylens_drone/core/capture.ts';
import { DroneApp } from '../../skylens_drone/core/drone.ts';
import { nodeSocketFactory } from '../../skylens_drone/node/run.ts';
import { manual, startFakeGateway } from './fakeGateway.ts';

const PORT = 8199;
const DRONE_ID = 1;

function arg(name: string, fallback: number): number {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? Number(hit.split('=')[1]) : fallback;
}

async function run(): Promise<void> {
  const seconds = arg('seconds', 55);
  const mode: LinkMode = process.argv.includes('--mode=webrtc') ? 'webrtc' : 'relay';
  const started = Date.now();
  const stamp = () => `+${((Date.now() - started) / 1000).toFixed(1).padStart(5)}s`;
  const say = (line: string) => console.log(`${stamp()} ${line}`);

  let telemetryCount = 0;
  let lastTelemetryPrint = 0;
  let segmentCount = 0;

  const gateway = await startFakeGateway({
    port: PORT,
    mode,
    routeDelayMs: 2000,
    onLog: (line) => say(line),
    onFrame: (env: Envelope<UplinkMessage>, via) => {
      const p = env.payload;
      if (p.kind === 'drone-hello') {
        say(
          `[gateway/${via}] <- drone-hello    seq=${env.seq} id=${p.droneId} ` +
            `mode=${p.mode} model="${p.model}"`,
        );
      } else if (p.kind === 'telemetry') {
        telemetryCount++;
        const now = Date.now();
        if (now - lastTelemetryPrint > 2000) {
          lastTelemetryPrint = now;
          say(
            `[gateway/${via}] <- telemetry     seq=${env.seq} #${telemetryCount} ` +
              `lat=${p.gps.lat.toFixed(5)} lon=${p.gps.lon.toFixed(5)} alt=${p.gps.alt.toFixed(1)} ` +
              `hdg=${String(p.headingDeg).padStart(5)} spd=${p.speed} batt=${p.batteryPct}%`,
          );
        }
      } else if (p.kind === 'video-segment') {
        segmentCount++;
        const first = p.poses[0];
        const last = p.poses[p.poses.length - 1];
        say(
          `[gateway/${via}] <- VIDEO-SEGMENT  seq=${env.seq} slice#${p.seq} codec=${p.codec} ` +
            `uri=${p.uri} bytes=${p.bytes} dur=${p.durationMs}ms poses=${p.poses.length} ` +
            `(first ${first?.gps.lat.toFixed(5)},${first?.gps.lon.toFixed(5)} -> ` +
            `last ${last?.gps.lat.toFixed(5)},${last?.gps.lon.toFixed(5)})`,
        );
      }
    },
  });

  const app = new DroneApp({
    config: {
      ...DEFAULT_CONFIG,
      droneId: DRONE_ID,
      demo: true,
      mode,
      gatewayUrl: gatewayUrlFor(mode, `ws://127.0.0.1:${PORT}`),
      telemetryHz: 5,
      slicesPerLeg: 4,
      arrivalMs: 10_000,
      cruiseSpeed: 12,
    },
    capture: new DemoCapture(),
    socketFactory: nodeSocketFactory,
    onLog: (line) => say(`[drone]                    ${line}`),
  });
  await app.start();

  // Manual takeover probe near the end, then release.
  setTimeout(() => {
    say('[gateway] -> manual-control forward=1 yaw=0.4 (operator takes over)');
    gateway.send(manual(DRONE_ID, 1, 0.4));
  }, (seconds - 12) * 1000);
  setTimeout(() => {
    say('[gateway] -> manual-control 0,0,0 (sticks released)');
    gateway.send(manual(DRONE_ID, 0, 0));
  }, (seconds - 9) * 1000);

  setTimeout(() => {
    const snap = app.snapshot;
    say('--- summary -------------------------------------------------');
    say(`  phase=${snap.phase} note=${snap.note}`);
    say(
      `  link: ${snap.link.phase} carrier=${snap.link.carrier} mode=${snap.link.mode} ` +
        `sent=${snap.link.sent} dropped=${snap.link.dropped}` +
        (snap.link.directUrl ? ` direct=${snap.link.directUrl}` : ''),
    );
    say(
      `  flown=${snap.odometerM.toFixed(0)} m of a ${snap.routeLengthM.toFixed(0)} m route, ` +
        `lap=${snap.lap} dir=${snap.direction}`,
    );
    say(`  telemetry frames=${telemetryCount}  video segments=${segmentCount}`);
    say(`  gateway received ${gateway.frames.length} uplink frames total`);
    if (mode === 'webrtc') {
      say(`  media seen on the SIGNALLING socket: ${gateway.signalMedia} (must be 0)`);
    }
    app.stop();
    void gateway.close().then(() => process.exit(0));
  }, seconds * 1000);
}

void run();
