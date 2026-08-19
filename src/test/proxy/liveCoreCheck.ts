// Integration drill against the REAL core (src/skylens_core/server), not a stand-in.
//
//   npx tsx src/test/proxy/liveCoreCheck.ts
//
// It boots the core itself rather than trusting whatever holds port 8080, then
// puts a scripted drone through 게이트웨이 → 프록시 → 코어 and reads the core's own
// /health to confirm arrival. Finally it takes the core away mid-flight — the
// case that actually happens, since the core restarts whenever it is redeployed —
// and shows the proxy riding it out on the standby path and coming back.
//
// The core is started with SKYLENS_CORE_WEB_MODE=off: the control tower UI needs a
// Vite dev server, and this drill is about the transport, not the screen.

import { spawn } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const CORE = 'ws://127.0.0.1:8080/uplink';
const STANDBY = 'ws://127.0.0.1:8180/uplink';

interface Proc {
  label: string;
  child: ChildProcess;
  lines: string[];
}

const running: Proc[] = [];
const failures: string[] = [];

function start(label: string, script: string, args: string[], env: NodeJS.ProcessEnv = {}): Proc {
  const child = spawn(process.execPath, ['--import', 'tsx', script, ...args], {
    cwd: repoRoot,
    env: { ...process.env, ...env },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const proc: Proc = { label, child, lines: [] };
  const absorb = (buf: Buffer): void => {
    for (const line of buf.toString().split('\n')) {
      const text = line.trimEnd();
      if (text.length === 0) continue;
      proc.lines.push(text);
      console.log(`  ${label} | ${text}`);
    }
  };
  child.stdout?.on('data', absorb);
  child.stderr?.on('data', absorb);
  running.push(proc);
  return proc;
}

function stop(proc: Proc): void {
  const at = running.indexOf(proc);
  if (at >= 0) running.splice(at, 1);
  if (proc.child.exitCode === null) proc.child.kill('SIGKILL');
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function waitFor(ms: number, check: () => boolean): Promise<boolean> {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    if (check()) return true;
    await sleep(100);
  }
  return false;
}

function since(proc: Proc, mark: number): string[] {
  return proc.lines.slice(mark);
}

function check(name: string, ok: boolean, detail = ''): void {
  console.log(`[live] ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures.push(name);
}

interface CoreHealth {
  component?: string;
  uplink?: { sockets: number; socketsSeen: number; frames: number; rejected: number };
  drones?: Array<{ droneId: number; model: string; mode: string; slices: number }>;
  store?: { uplinkFrames?: number; lastSeq?: number };
}

async function coreHealth(): Promise<CoreHealth | null> {
  try {
    const res = await fetch('http://127.0.0.1:8080/health');
    return (await res.json()) as CoreHealth;
  } catch {
    return null;
  }
}

function startCore(): Proc {
  return start('core    ', 'src/skylens_core/server/index.ts', [], {
    SKYLENS_CORE_WEB_MODE: 'off',
  });
}

function drone(env: NodeJS.ProcessEnv): Promise<void> {
  const proc = start('drone   ', 'src/test/gateway/fakeDrone.ts', [], env);
  return new Promise((resolve) => proc.child.on('exit', () => resolve()));
}

async function main(): Promise<void> {
  console.log('[live] booting the real core, a standby path, proxy and gateway');
  let core = startCore();
  const standby = start('standby ', 'src/test/proxy/fakeCore.ts', ['8180', 'standby'], {
    FAKE_CORE_QUIET: '1',
  });
  const proxy = start('proxy   ', 'src/skylens_proxy/index.ts', [], {
    SKYLENS_CORE_ENDPOINTS: `${CORE},${STANDBY}`,
  });
  const gateway = start('gateway ', 'src/skylens_gateway/index.ts', []);

  const coreUp = await waitFor(30000, () => hasCoreListening(core));
  check('the real core is listening on 8080', coreUp);
  // The real core takes a couple of seconds to boot, so the standby is often
  // active first and the proxy fails BACK to the core — which is the design
  // working, not a fault: traffic has somewhere to go the whole time.
  const onCore = await waitFor(20000, () =>
    proxy.lines.some(
      (l) => l.includes(`active core path = ${CORE}`) || l.includes(`FAILBACK ${STANDBY} -> ${CORE}`),
    ),
  );
  check('proxy settled on the real core as its active path', onCore, CORE);

  const before = await coreHealth();
  await drone({ DRONE_FRAMES: '12', DRONE_TICK_MS: '300', DRONE_SEGMENT_EVERY: '3' });
  await sleep(1500);
  const after = await coreHealth();

  check('the core answers /health', after?.component === 'skylens_core');
  const framesBefore = before?.uplink?.frames ?? 0;
  const framesAfter = after?.uplink?.frames ?? 0;
  check(
    'the core counted the frames the proxy handed it',
    framesAfter > framesBefore,
    `uplink.frames ${framesBefore} → ${framesAfter}`,
  );
  const seenDrone = (after?.drones ?? []).find((d) => d.droneId === 1);
  check(
    'the core registered the drone behind two hops',
    seenDrone !== undefined,
    seenDrone ? `drone ${seenDrone.droneId} ${seenDrone.model} slices=${seenDrone.slices}` : 'none',
  );
  check('the core rejected nothing', (after?.uplink?.rejected ?? 0) === 0);

  console.log('\n[live] taking the core away mid-flight (a redeploy, not a disaster)');
  const proxyMark = proxy.lines.length;
  const standbyMark = standby.lines.length;
  stop(core);
  const failedOver = await waitFor(12000, () =>
    since(proxy, proxyMark).some((l) => l.includes(`FAILOVER ${CORE} -> ${STANDBY}`)),
  );
  check('proxy failed over while the core was gone', failedOver);
  await drone({ DRONE_ID: '3', DRONE_FRAMES: '6', DRONE_TICK_MS: '300', DRONE_SEGMENT_EVERY: '2' });
  check(
    'the drone kept being served with no core at all',
    since(standby, standbyMark).some((l) => l.includes('kind=video-segment')),
  );

  console.log('\n[live] core comes back');
  const backMark = proxy.lines.length;
  core = startCore();
  const failedBack = await waitFor(30000, () =>
    since(proxy, backMark).some((l) => l.includes(`FAILBACK ${STANDBY} -> ${CORE}`)),
  );
  check('proxy returned to the core once it was up again', failedBack);
  await drone({ DRONE_ID: '4', DRONE_FRAMES: '6', DRONE_TICK_MS: '300', DRONE_SEGMENT_EVERY: '2' });
  await sleep(1000);
  const final = await coreHealth();
  check(
    'the restarted core is receiving again',
    (final?.uplink?.frames ?? 0) > 0,
    `uplink.frames=${final?.uplink?.frames ?? 0}`,
  );
  void gateway;
}

function hasCoreListening(core: Proc): boolean {
  return core.lines.some((l) => l.includes('uplink  ws://'));
}

main()
  .catch((err) => {
    console.error(`[live] crashed: ${String(err)}`);
    failures.push('drill crashed');
  })
  .finally(async () => {
    for (const proc of [...running]) stop(proc);
    await sleep(300);
    console.log(
      failures.length === 0 ? '\n[live] ALL CHECKS PASSED' : `\n[live] FAILED: ${failures.join(', ')}`,
    );
    process.exit(failures.length === 0 ? 0 : 1);
  });
