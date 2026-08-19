// Drives the demo scenario end to end without a human at the control tower.
//
//   npm run demo                                   (in one shell)
//   node --import tsx src/test/demo/driveScenario.ts   (in another)
//
// It speaks the control tower's half of the contract — one `assign-route` on
// the core's viewer socket — and then just listens, printing the phases and the
// delay-pattern deliveries as they arrive. Everything it reports is the core's
// own output; nothing here simulates a component.
//
// What it proves, if it passes: the scenario in COMPONENTS.md §5.2 actually
// happens, in order, driven by drone movement rather than a timer here.

import process from 'node:process';
import { WebSocket } from 'ws';
import { DEMO_ROUTE } from '../../skylens_drone/core/config.ts';
import type { Envelope, ViewerMessage } from '../../shared/protocol.ts';

const CORE = process.env.SKYLENS_CORE_VIEWER ?? 'ws://127.0.0.1:8080/viewer';
const WATCH_MS = Number(process.env.SKYLENS_SCENARIO_MS ?? 150_000);

const started = Date.now();
const t = (): string => `t+${((Date.now() - started) / 1000).toFixed(1)}s`;

/** Highest level delivered per segment, in arrival order. */
const levels = new Map<number, number>();
const phases: string[] = [];
let overlapSeen: string | null = null;
let detections = 0;

function unwrap(raw: string): ViewerMessage | null {
  try {
    const parsed = JSON.parse(raw) as Partial<Envelope<ViewerMessage>> & Partial<ViewerMessage>;
    if (typeof parsed.kind === 'string') return parsed as ViewerMessage;
    const payload = (parsed as Envelope<ViewerMessage>).payload;
    return payload && typeof payload.kind === 'string' ? payload : null;
  } catch {
    return null;
  }
}

const ws = new WebSocket(CORE);

ws.on('open', () => {
  console.log(`${t()} connected to ${CORE}`);
  const cmd = {
    kind: 'assign-route' as const,
    droneId: 1,
    waypoints: DEMO_ROUTE,
    // COMPONENTS.md §5.2 step 5: fly the route back and forth.
    loop: true,
  };
  ws.send(JSON.stringify(cmd));
  console.log(`${t()} assign-route sent — ${DEMO_ROUTE.length} waypoints, loop=true`);
});

ws.on('message', (data: Buffer) => {
  const msg = unwrap(data.toString());
  if (!msg) return;

  if (msg.kind === 'mission-status') {
    const line = `${msg.phase}${msg.etaSeconds !== null ? ` eta=${msg.etaSeconds}` : ''} · ${msg.message}`;
    if (phases[phases.length - 1] !== msg.phase) {
      phases.push(msg.phase);
      console.log(`${t()} MISSION ${line}`);
    }
    return;
  }

  if (msg.kind === 'splat-chunk') {
    const prev = levels.get(msg.segment) ?? 0;
    levels.set(msg.segment, msg.level);
    console.log(
      `${t()} CHUNK   segment ${msg.segment} level ${msg.level} (${msg.steps} steps, ${(msg.bytes / 1024).toFixed(0)} KiB)` +
        `${prev > 0 ? ` — replaces level ${prev}` : ''}${msg.final ? ' FINAL' : ''}`,
    );
    // The delay pattern's whole claim: an earlier segment is still climbing its
    // ladder while a later one delivers its first level.
    if (!overlapSeen) {
      for (const [seg, lv] of levels) {
        if (seg < msg.segment && lv > 1 && msg.level === 1) {
          overlapSeen = `segment ${seg} at level ${lv} while segment ${msg.segment} delivered level 1`;
          console.log(`${t()} OVERLAP ${overlapSeen}`);
        }
      }
    }
    return;
  }

  if (msg.kind === 'detection') {
    detections += 1;
    console.log(`${t()} DETECT  ${msg.category} in segment ${msg.segment} (${msg.label})`);
  }
});

ws.on('error', (err: Error) => {
  console.error(`${t()} socket error: ${err.message}`);
  process.exitCode = 1;
});

setTimeout(() => {
  console.log('');
  console.log('--- scenario summary ---');
  console.log(`phases       : ${phases.join(' → ')}`);
  console.log(`segments     : ${[...levels.keys()].sort((a, b) => a - b).join(', ') || '(none)'}`);
  console.log(`levels       : ${[...levels.entries()].map(([s, l]) => `s${s}:L${l}`).join(' ')}`);
  console.log(`detections   : ${detections}`);
  console.log(`overlap      : ${overlapSeen ?? 'NOT OBSERVED'}`);

  const ok =
    phases.join(',') === 'idle,assigned,awaiting-drone,active' ||
    phases.join(',') === 'assigned,awaiting-drone,active';
  console.log(`phase order  : ${ok ? 'as specified' : 'UNEXPECTED'}`);
  console.log(`result       : ${ok && overlapSeen && levels.size > 1 ? 'PASS' : 'FAIL'}`);
  ws.close();
  process.exit(ok && overlapSeen && levels.size > 1 ? 0 : 1);
}, WATCH_MS);
