// Headless drone runner.
//
//   npx tsx src/skylens_drone/node/run.ts --demo --gateway=ws://127.0.0.1:8081/drone
//
// This is the entry point skylens_demo launches: it is the exact same DroneApp
// the Tauri window runs, with the browser's WebSocket swapped for `ws` and the
// operator panel replaced by stdout. No Rust build required.

import { existsSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { WebSocket } from 'ws';
import { DEMO_ROUTE, envFromArgv, resolveConfig } from '../core/config.ts';
import { DemoCapture } from '../core/capture.ts';
import { DEMO_CLIPS } from '../core/demoAssets.ts';
import { DroneApp } from '../core/drone.ts';
import type { SocketLike } from '../core/link.ts';

export function nodeSocketFactory(url: string): SocketLike {
  return new WebSocket(url) as unknown as SocketLike;
}

/**
 * The demo footage is GENERATED (H.265 transcodes of the recorded H.264 clips),
 * so a fresh clone does not have it. Fail here with the command to run rather
 * than mid-flight with a broken uri — or, worse, by shipping segments that point
 * at nothing.
 */
export function checkDemoAssets(): string[] {
  const missing: string[] = [];
  for (const clip of DEMO_CLIPS) {
    const file = path.resolve(clip.uri.replace(/^\//, ''));
    if (!existsSync(file)) missing.push(file);
  }
  return missing;
}

export async function main(argv: string[]): Promise<DroneApp> {
  const config = resolveConfig(process.env, envFromArgv(argv));
  if (!config.demo) {
    console.error(
      '[drone] the headless runner only supports demo capture (there is no camera in Node).\n' +
        '        run with --demo / SKYLENS_DEMO=1, or use the Tauri app for a live capture.',
    );
    process.exit(1);
  }

  const missing = checkDemoAssets();
  if (missing.length > 0) {
    console.error(
      `[drone] ${missing.length} demo clip(s) missing, e.g. ${missing[0]}\n` +
        '        the footage is transcoded to H.265 once, run:\n' +
        '          npx tsx src/skylens_drone/tools/transcodeDemoFootage.ts',
    );
    process.exit(1);
  }

  const started = Date.now();
  const stamp = () => `+${((Date.now() - started) / 1000).toFixed(1)}s`;

  const app = new DroneApp({
    config,
    capture: new DemoCapture(),
    socketFactory: nodeSocketFactory,
    onLog: (line) => console.log(`[drone ${stamp()}] ${line}`),
  });

  console.log(
    `[drone] id=${config.droneId} mode=${config.mode} gateway=${config.gatewayUrl} ` +
      `slices/leg=${config.slicesPerLeg} telemetry=${config.telemetryHz}Hz arrival=${config.arrivalMs}ms`,
  );
  await app.start();

  if (config.autoRoute) {
    app.assignRoute({ kind: 'assign-route', droneId: config.droneId, waypoints: DEMO_ROUTE, loop: true });
  }

  const shutdown = () => {
    app.stop();
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
  return app;
}

const invoked = process.argv[1] ?? '';
if (invoked.replace(/\\/g, '/').endsWith('skylens_drone/node/run.ts')) {
  void main(process.argv.slice(2));
}
