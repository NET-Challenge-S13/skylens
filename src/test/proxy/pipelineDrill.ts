// Transport drill: 드론 → 게이트웨이 → 프록시 → 코어, plus the failover the proxy
// exists for. Runs the real component entry points as child processes and asserts
// on what they actually print and on their /health output.
//
//   npx tsx src/test/proxy/pipelineDrill.ts
//
// Phases
//   1  relay: a scripted drone pushes telemetry + a VideoSegment, and the frame is
//      observed arriving at the PRIMARY core carrying the gateway→proxy hop stamps.
//   2  the primary core is killed: FAILOVER to the standby, traffic keeps landing.
//   3  the primary comes back: FAILBACK.
//   4  the primary hangs (socket open, no pongs): FAILOVER on health-probe timeout
//      alone, then FAILBACK when it answers again.
//   5  webrtc: the gateway brokers a punch and the drone's media reaches the core
//      with NO gateway hop in the path — the mode's whole claim.
//
// Exit code 0 = every phase passed. Nothing here is collected by Playwright or
// pytest; it is a harness you run when you touch the transport.

import { spawn } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

const PRIMARY = 'ws://127.0.0.1:8080/uplink';
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

/** Poll until `check` is true, or give up. */
async function waitFor(what: string, ms: number, check: () => boolean): Promise<boolean> {
  const until = Date.now() + ms;
  while (Date.now() < until) {
    if (check()) return true;
    await sleep(100);
  }
  return false;
}

function has(proc: Proc, needle: string): boolean {
  return proc.lines.some((l) => l.includes(needle));
}

/** Lines a proc printed after a mark, so a phase never matches an older event. */
function since(proc: Proc, mark: number): string[] {
  return proc.lines.slice(mark);
}

function check(name: string, ok: boolean, detail = ''): void {
  const tag = ok ? 'PASS' : 'FAIL';
  console.log(`[drill] ${tag}  ${name}${detail ? ` — ${detail}` : ''}`);
  if (!ok) failures.push(name);
}

async function health(port: number): Promise<Record<string, unknown> | null> {
  try {
    const res = await fetch(`http://127.0.0.1:${port}/health`);
    return (await res.json()) as Record<string, unknown>;
  } catch {
    return null;
  }
}

/** Run a scripted drone to completion. */
function drone(script: string, args: string[], env: NodeJS.ProcessEnv): Promise<Proc> {
  const proc = start('drone   ', script, args, env);
  return new Promise((resolve) => proc.child.on('exit', () => resolve(proc)));
}

function startPrimary(env: NodeJS.ProcessEnv = {}): Proc {
  return start('primary ', 'src/test/proxy/fakeCore.ts', ['8080', 'primary'], {
    FAKE_CORE_QUIET: '1',
    ...env,
  });
}

async function main(): Promise<void> {
  console.log('[drill] booting standby core, primary core, proxy, gateway(relay)');
  const standby = start('standby ', 'src/test/proxy/fakeCore.ts', ['8180', 'standby'], {
    FAKE_CORE_QUIET: '1',
  });
  let primary = startPrimary();
  const proxy = start('proxy   ', 'src/skylens_proxy/index.ts', [], {
    SKYLENS_CORE_ENDPOINTS: `${PRIMARY},${STANDBY}`,
  });
  let gateway = start('gateway ', 'src/skylens_gateway/index.ts', []);

  const up = await waitFor(
    'both /health answering',
    15000,
    () => has(proxy, 'health: http') && has(gateway, 'health: http'),
  );
  check('proxy and gateway came up', up);
  const active = await waitFor('active path chosen', 10000, () =>
    has(proxy, `active core path = ${PRIMARY}`),
  );
  check('primary chosen as the active KOREN path', active, PRIMARY);

  // -- phase 1: relay ------------------------------------------------------
  console.log('\n[drill] phase 1 — telemetry + VideoSegment through gateway → proxy → core');
  await drone('src/test/gateway/fakeDrone.ts', [], {
    DRONE_FRAMES: '9',
    DRONE_TICK_MS: '300',
    DRONE_SEGMENT_EVERY: '3',
  });
  const landed = await waitFor('segment at primary', 5000, () =>
    primary.lines.some((l) => l.includes('kind=video-segment') && l.includes('gateway->proxy')),
  );
  check('VideoSegment reached the core with both hop stamps', landed);
  check(
    'telemetry reached the core',
    primary.lines.some((l) => l.includes('kind=telemetry')),
  );
  check(
    'the core sees which KOREN path carried it',
    primary.lines.some((l) => l.includes(`proxy(${PRIMARY})`)),
  );

  // -- phase 2: failover ---------------------------------------------------
  console.log('\n[drill] phase 2 — killing the primary core');
  const standbyMark = standby.lines.length;
  let proxyMark = proxy.lines.length;
  stop(primary);
  const failedOver = await waitFor('failover', 10000, () =>
    since(proxy, proxyMark).some((l) => l.includes(`FAILOVER ${PRIMARY} -> ${STANDBY}`)),
  );
  check('proxy failed over to the standby path', failedOver);

  await drone('src/test/gateway/fakeDrone.ts', [], {
    DRONE_ID: '2',
    DRONE_FRAMES: '6',
    DRONE_TICK_MS: '300',
    DRONE_SEGMENT_EVERY: '2',
  });
  const onStandby = await waitFor('segment at standby', 5000, () =>
    since(standby, standbyMark).some((l) => l.includes('kind=video-segment')),
  );
  check('traffic kept flowing over the standby path', onStandby);

  // -- phase 3: failback ---------------------------------------------------
  console.log('\n[drill] phase 3 — primary core comes back');
  proxyMark = proxy.lines.length;
  primary = startPrimary();
  const failedBack = await waitFor('failback', 12000, () =>
    since(proxy, proxyMark).some((l) => l.includes(`FAILBACK ${STANDBY} -> ${PRIMARY}`)),
  );
  check('proxy failed back to the better path', failedBack);

  // -- phase 4: the far end stops answering (socket still open) ------------
  console.log('\n[drill] phase 4 — primary stops answering pings without closing');
  stop(primary);
  // Let the port come free before the replacement binds it.
  await sleep(1000);
  proxyMark = proxy.lines.length;
  primary = startPrimary({ FAKE_CORE_HANG_AFTER_MS: '4000', FAKE_CORE_HANG_FOR_MS: '7000' });
  await waitFor('back on primary', 15000, () =>
    since(proxy, proxyMark).some((l) => l.includes(`-> ${PRIMARY}`)),
  );
  proxyMark = proxy.lines.length;
  const hungOut = await waitFor('failover on silence', 20000, () =>
    since(proxy, proxyMark).some((l) => l.includes(`FAILOVER ${PRIMARY} -> ${STANDBY}`)),
  );
  check('health probe caught a live socket with a dead far end', hungOut);
  check(
    'the reason was silence, not a close',
    since(proxy, proxyMark).some((l) => l.includes('UNHEALTHY') && l.includes('no response')),
  );
  const recovered = await waitFor('recovery', 20000, () =>
    since(proxy, proxyMark).some((l) => l.includes(`FAILBACK ${STANDBY} -> ${PRIMARY}`)),
  );
  check('recovered to the primary once it answered again', recovered);

  // -- phase 5: webrtc mode ------------------------------------------------
  console.log('\n[drill] phase 5 — webrtc mode: gateway brokers, carries no media');
  stop(gateway);
  await sleep(500);
  gateway = start('gateway ', 'src/skylens_gateway/index.ts', [], {
    SKYLENS_GATEWAY_MODE: 'webrtc',
  });
  await waitFor('gateway(webrtc) up', 15000, () => has(gateway, 'brokering hole punching'));
  const primaryMark = primary.lines.length;
  await drone('src/test/gateway/fakeDroneWebrtc.ts', [], {
    DRONE_ID: '7',
    DRONE_FRAMES: '6',
    DRONE_TICK_MS: '300',
  });
  check(
    'gateway reported the punch and stepped out',
    has(gateway, 'punched through'),
  );
  const direct = await waitFor('direct media at core', 6000, () =>
    since(primary, primaryMark).some(
      (l) => l.includes('kind=video-segment') && !l.includes('gateway'),
    ),
  );
  check('drone media reached the core with NO gateway hop', direct);

  const gwHealth = await health(8081);
  const counters = (gwHealth?.counters ?? {}) as Record<string, number>;
  check(
    'gateway /health reports signalling sessions only',
    gwHealth?.mode === 'webrtc' && counters.sessionsEstablished >= 1,
    `sessionsEstablished=${counters.sessionsEstablished}`,
  );
  const pxHealth = await health(8082);
  const ingress = (pxHealth?.ingress ?? {}) as Record<string, number | string>;
  check(
    'proxy /health accounts for the direct drone ingress',
    Number(ingress.seenDroneDirect) >= 1 && Number(ingress.framesFromDroneDirect) >= 1,
    JSON.stringify(ingress),
  );
}

main()
  .catch((err) => {
    console.error(`[drill] crashed: ${String(err)}`);
    failures.push('drill crashed');
  })
  .finally(async () => {
    for (const proc of [...running]) stop(proc);
    await sleep(300);
    console.log(
      failures.length === 0
        ? '\n[drill] ALL PHASES PASSED'
        : `\n[drill] FAILED: ${failures.join(', ')}`,
    );
    process.exit(failures.length === 0 ? 0 : 1);
  });
