// Demo launcher — assembles the whole pipeline on one machine.
//
//   npm run demo
//
// The components are the real ones. Nothing here reimplements a component's
// behaviour; the launcher only starts them in dependency order with demo
// settings and keeps their output legible. What "demo mode" swaps is stated in
// COMPONENTS.md §5.1: the drone plays recorded footage instead of a camera, and
// the model API resolves pre-built segment assets instead of training.
//
// Startup order matters. Each process is health-gated before the next starts,
// because a component that comes up without its upstream spends its first
// seconds retrying and floods the log with failures that look like bugs:
//
//   model(8100) -> core(8080) -> proxy(8082) -> gateway(8081) -> client(8090)
//                                                                 -> vite(5173)
//                                                                 -> drone
//
// The drone comes last and stays idle: the scenario starts when an operator
// assigns a route in the control tower (COMPONENTS.md §5.2).

import { spawn, spawnSync } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import net from 'node:net';
import process from 'node:process';
import readline from 'node:readline';

interface Component {
  name: string;
  /** Shown in the log gutter; kept short so lines align. */
  tag: string;
  command: string;
  args: string[];
  env: Record<string, string>;
  /** Polled until it answers before the next component starts. Null = no gate. */
  health: string | null;
  /** Give up waiting after this long. */
  readyTimeoutMs: number;
}

const NODE = process.execPath;
/** Run a TypeScript entry point without a build step, the way the READMEs do. */
const tsx = (entry: string): string[] => ['--import', 'tsx', entry];

const DEMO_ENV: Record<string, string> = {
  SKYLENS_DEMO: '1',
};

const COMPONENTS: Component[] = [
  {
    name: 'skylens_model',
    tag: 'model ',
    command: 'uv',
    args: ['run', 'uvicorn', 'skylens_model.app:app', '--port', '8100', '--log-level', 'warning'],
    env: { ...DEMO_ENV },
    health: 'http://127.0.0.1:8100/health',
    // uv resolves the environment on first run, which can take a while.
    readyTimeoutMs: 120_000,
  },
  {
    name: 'skylens_core',
    tag: 'core  ',
    command: NODE,
    args: tsx('src/skylens_core/server/index.ts'),
    env: {
      ...DEMO_ENV,
      SKYLENS_CORE_WEB_MODE: 'dev',
      // The demo flies in Daejeon (src/test/geography.spec.ts pins the anchor,
      // the tower's map and the drone waypoints together), so the planner opens
      // there rather than wherever this machine's ISP is registered.
      SKYLENS_CORE_SITE: '36.3685,127.3475,30',
    },
    health: 'http://127.0.0.1:8080/health',
    readyTimeoutMs: 30_000,
  },
  {
    name: 'skylens_proxy',
    tag: 'proxy ',
    command: NODE,
    args: tsx('src/skylens_proxy/index.ts'),
    env: { ...DEMO_ENV },
    health: 'http://127.0.0.1:8082/health',
    readyTimeoutMs: 20_000,
  },
  {
    name: 'skylens_gateway',
    tag: 'gatewy',
    command: NODE,
    args: tsx('src/skylens_gateway/index.ts'),
    env: { ...DEMO_ENV, SKYLENS_GATEWAY_MODE: 'relay' },
    health: 'http://127.0.0.1:8081/health',
    readyTimeoutMs: 20_000,
  },
  {
    name: 'skylens_client',
    tag: 'client',
    command: NODE,
    args: tsx('src/skylens_client/server/index.ts'),
    env: { ...DEMO_ENV },
    health: 'http://127.0.0.1:8090/health',
    readyTimeoutMs: 20_000,
  },
  {
    name: 'vite',
    tag: 'vite  ',
    command: NODE,
    args: ['node_modules/vite/bin/vite.js'],
    env: {},
    // Both web servers proxy to Vite in dev mode, so it has to be up before a
    // browser hits either of them.
    health: 'http://127.0.0.1:5173/res/static/index.html',
    readyTimeoutMs: 60_000,
  },
  // The formation: three aircraft abeam, named for where they fly (중간보고서
  // Ⅲ-1-나). Each is its own drone process — that is what a drone is — and all
  // three fly the same assigned route, offsetting into their station. All three
  // film, so the operator can switch the camera panel between them.
  //
  // Ids are the routing addresses; the operator never sees them. The centre
  // aircraft is id 1 because a route is assigned to the formation through it.
  ...([
    { station: 'center', id: 1 },
    { station: 'left', id: 2 },
    { station: 'right', id: 3 },
  ] as const).map(({ station, id }) => ({
    name: `skylens_drone(${station})`,
    tag: station.padEnd(6).slice(0, 6),
    command: NODE,
    args: [...tsx('src/skylens_drone/node/run.ts'), '--demo'],
    // The drone processes are up from the start, but must not announce
    // themselves until they have "reached the site" — otherwise the core sees a
    // drone online immediately and the mission jumps straight from assigned to
    // active, skipping the drone-connection wait the scenario is built around
    // (COMPONENTS.md §5.2 steps 3-4).
    env: {
      ...DEMO_ENV,
      SKYLENS_DRONE_HELLO_ON_ARRIVAL: '1',
      SKYLENS_DRONE_ID: String(id),
      SKYLENS_DRONE_STATION: station,
      SKYLENS_DRONE_MODEL: `SkyLens D-${station.toUpperCase()}`,
    },
    health: null,
    readyTimeoutMs: 0,
  })),
];

const running: Array<{ name: string; child: ChildProcess }> = [];
let shuttingDown = false;

/** The port a component's health URL names, or null when it is not gated. */
function healthPort(component: Component): number | null {
  if (!component.health) return null;
  const port = Number(new URL(component.health).port);
  return Number.isFinite(port) && port > 0 ? port : null;
}

/** Is something already listening here? */
function portTaken(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const probe = net.connect({ host: '127.0.0.1', port });
    const done = (taken: boolean): void => {
      probe.destroy();
      resolve(taken);
    };
    probe.setTimeout(700);
    probe.on('connect', () => done(true));
    probe.on('timeout', () => done(false));
    probe.on('error', () => done(false));
  });
}

/**
 * Refuse to start on top of a previous run.
 *
 * waitFor() only asks whether SOMETHING answers the health URL, and a survivor
 * from an earlier run answers it perfectly — so the launcher would report the
 * component ready while the demo quietly ran against the old code. That looks
 * exactly like a caching bug and costs an afternoon to see through. Check the
 * ports first and say who is holding them.
 */
async function preflight(): Promise<void> {
  const taken: number[] = [];
  for (const component of COMPONENTS) {
    const port = healthPort(component);
    if (port !== null && (await portTaken(port))) taken.push(port);
  }
  if (taken.length === 0) return;
  console.error('');
  console.error(`  포트 ${taken.join(', ')} 를 이미 다른 프로세스가 쓰고 있습니다.`);
  console.error('  이전 데모가 완전히 종료되지 않았을 가능성이 큽니다 — 그대로 두면');
  console.error('  런처가 그 프로세스를 자기 컴포넌트로 오인해 옛 코드로 데모가 돕니다.');
  console.error('');
  console.error('  정리 후 다시 실행하십시오:');
  console.error('    Windows  npm run demo:clean');
  console.error("    그 외     lsof -ti:" + taken.join(',') + ' | xargs kill -9');
  console.error('');
  process.exit(1);
}

/**
 * Kill a child and everything it spawned.
 *
 * `uv run uvicorn` is a launcher of launchers: killing the child leaves the
 * python server alive, still holding its port, ready to be mistaken for the
 * next run's component. Take the whole tree down.
 */
function killTree(child: ChildProcess): void {
  if (child.exitCode !== null || child.pid === undefined) return;
  try {
    if (process.platform === 'win32') {
      spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
    } else {
      process.kill(-child.pid, 'SIGTERM');
    }
  } catch {
    /* already gone */
  }
  try {
    child.kill();
  } catch {
    /* already gone */
  }
}

function stamp(): string {
  const d = new Date();
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
}

function pipe(tag: string, child: ChildProcess): void {
  for (const stream of [child.stdout, child.stderr]) {
    if (!stream) continue;
    readline.createInterface({ input: stream }).on('line', (line) => {
      if (line.trim()) console.log(`${stamp()} ${tag} | ${line}`);
    });
  }
}

async function reachable(url: string): Promise<boolean> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(1500) });
    return res.ok;
  } catch {
    return false;
  }
}

async function waitFor(component: Component): Promise<void> {
  if (!component.health) return;
  const deadline = Date.now() + component.readyTimeoutMs;
  while (Date.now() < deadline) {
    if (shuttingDown) return;
    if (await reachable(component.health)) {
      console.log(`${stamp()} launch | ${component.name} ready (${component.health})`);
      return;
    }
    await new Promise((r) => setTimeout(r, 400));
  }
  throw new Error(
    `${component.name} did not answer ${component.health} within ${component.readyTimeoutMs / 1000}s`,
  );
}

function start(component: Component): void {
  const child = spawn(component.command, component.args, {
    cwd: process.cwd(),
    env: { ...process.env, ...component.env },
    stdio: ['ignore', 'pipe', 'pipe'],
    // `uv` is a shim on Windows, so it needs the shell to be resolved. Node is
    // an absolute path and must NOT go through the shell (its args would be
    // re-parsed).
    shell: component.command === 'uv',
    // A process group of its own, so shutdown can signal the whole tree.
    // Windows has no groups; killTree uses taskkill /T there instead.
    detached: process.platform !== 'win32',
  });
  running.push({ name: component.name, child });
  pipe(component.tag, child);

  child.on('exit', (code, signal) => {
    if (shuttingDown) return;
    console.log(`${stamp()} launch | ${component.name} exited (code=${code} signal=${signal})`);
    // One component dying leaves the rest talking to a hole. Stop the whole
    // demo instead of pretending the pipeline is intact.
    void shutdown(code ?? 1);
  });
}

async function shutdown(code: number): Promise<void> {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`${stamp()} launch | stopping ${running.length} component(s)`);
  for (const { child } of [...running].reverse()) killTree(child);
  // Give them a moment to close their sockets before the process image goes.
  await new Promise((r) => setTimeout(r, 700));
  process.exit(code);
}

function banner(): void {
  console.log('');
  console.log('  SkyLens 데모');
  console.log('  ─────────────────────────────────────────────────────────────');
  console.log('  관제탑   http://localhost:8080/res/static/control.html');
  console.log('  현황판   http://localhost:8090/res/static/status.html');
  console.log('  드론     http://localhost:5173/src/skylens_drone/index.html');
  console.log('');
  console.log('  시나리오');
  console.log('   1. 지금은 정지 상태 — 드론은 대기, 복원 없음');
  console.log('   2. 관제탑에서 경로 계획으로 경유 지점을 지정하고 배정');
  console.log('   3. "태스크 지정 완료" 표기 후 드론 연결 대기 (약 10초)');
  console.log('   4. 드론이 연결되면 지정 경로를 왕복 비행');
  console.log('   5. 드론이 구간을 지날 때마다 현황판이 딜레이 패턴으로 갱신');
  console.log('');
  console.log('  Ctrl+C 로 전체 종료');
  console.log('');
}

async function main(): Promise<void> {
  process.on('SIGINT', () => void shutdown(0));
  process.on('SIGTERM', () => void shutdown(0));

  await preflight();

  for (const component of COMPONENTS) {
    console.log(`${stamp()} launch | starting ${component.name}`);
    start(component);
    await waitFor(component);
    if (shuttingDown) return;
  }
  banner();
}

main().catch((err: unknown) => {
  console.error(`${stamp()} launch | ${err instanceof Error ? err.message : String(err)}`);
  void shutdown(1);
});
