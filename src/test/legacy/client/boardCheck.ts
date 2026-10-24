/// <reference types="node" />
// End-to-end check for skylens_client: the relay server, the board, and the
// contract between them. Not a Playwright *spec* on purpose — it drives a real
// browser through a scripted timeline and PRINTS what it observed, because the
// things being checked here are temporal ("does segment 2 appear only once its
// chunk lands?") and a pass/fail line would hide the evidence.
//
// Assumes already running:  vite dev (5173)  ·  client relay (8090)
// Spawns and kills itself:  the fake core    (8080, src/test/client/fakeCore.ts)
//
// Run:  npx tsx src/test/client/boardCheck.ts

import { spawn, spawnSync } from 'node:child_process';
import type { ChildProcess } from 'node:child_process';
import WebSocket from 'ws';
import { chromium } from '@playwright/test';
import type { Browser, Page } from '@playwright/test';

const BOARD = 'http://localhost:8090/res/static/status.html';
const HEALTH = 'http://localhost:8090/health';

interface Probe {
  hasGeometry: boolean;
  splatStatus: string;
  chunks: number;
  replaced: number;
  segmentLevels: Record<number, number>;
  scenes: string[];
  relay: string;
  upstream: string;
  detail: string;
  receiving: boolean;
  serverChunks: number;
  serverDetections: number;
  segments: Array<{ index: number; level: number; levels: number; label: string }>;
  mission: string | null;
  markers: Array<{ id: string; segment: number; visible: boolean }>;
  banner: string;
  drones: number;
}

const PROBE = `(() => {
  const s = window.skylens;
  const banner = document.getElementById('board-waiting');
  const feed = s.server;
  return {
    hasGeometry: s.splat.hasGeometry,
    splatStatus: s.splat.status,
    chunks: s.splat.chunks,
    replaced: s.splat.replaced,
    segmentLevels: s.splat.segmentLevels,
    scenes: s.dbg.scenes,
    relay: feed.relay,
    upstream: feed.upstream,
    detail: feed.detail,
    receiving: feed.receiving,
    serverChunks: feed.server.chunks,
    serverDetections: feed.server.detections,
    segments: feed.server.segments,
    mission: feed.mission ? feed.mission.message : null,
    markers: s.markers,
    banner: banner ? banner.textContent : '',
    drones: s.state.drones.length,
  };
})()`;

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** Vite's HMR client reloads the page when the dev server blips, which would
 *  otherwise abort the run mid-timeline. Re-wait for the handle and carry on. */
async function probe(page: Page): Promise<Probe> {
  try {
    return (await page.evaluate(PROBE)) as Probe;
  } catch {
    await page.waitForFunction('window.skylens && window.skylens.splat', null, { timeout: 90_000 });
    return (await page.evaluate(PROBE)) as Probe;
  }
}

function line(t: number, p: Probe): string {
  const ladder = p.segments
    .map((s) => `${s.index}:${s.level}/${s.levels}`)
    .join(' ');
  const shown = p.markers.filter((m) => m.visible).map((m) => m.id).join(',') || '-';
  return (
    `t+${t.toString().padStart(2, ' ')}s  relay=${p.relay} core=${p.upstream}` +
    `  geom=${p.hasGeometry ? 'Y' : 'N'} scenes=[${p.scenes.join(' ')}]` +
    `  chunks=${p.chunks} replaced=${p.replaced} drones=${p.drones}` +
    `  ladder=[${ladder}] markers=${shown}`
  );
}

async function health(): Promise<Record<string, unknown>> {
  const res = await fetch(HEALTH);
  return (await res.json()) as Record<string, unknown>;
}

/** `shell: true` on Windows puts cmd.exe between us and node, so kill() would
 *  only take out the shell and leave port 8080 held. Kill the whole tree. */
function stopCore(child: ChildProcess): void {
  if (process.platform === 'win32' && child.pid) {
    spawnSync('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
  } else {
    child.kill();
  }
}

function startCore(): ChildProcess {
  const child = spawn('npx', ['tsx', 'src/test/client/fakeCore.ts'], {
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: process.platform === 'win32',
    env: {
      ...process.env,
      FAKE_FIRST: '1',
      FAKE_PERIOD: '4',
      FAKE_LEVEL_DELAYS: '0,2,6',
      // Levels 1-3 (250/1000/3500 steps). Level 4 is a 6.5 MB PLY per segment
      // and the check runs on SwiftShader, where the load — not the protocol —
      // becomes the bottleneck. The replacement rule is the same at every rung.
      FAKE_MAX_LEVEL: process.env.FAKE_MAX_LEVEL ?? '3',
    },
  });
  child.stdout?.on('data', (b: Buffer) => process.stdout.write(`   ${b}`));
  child.stderr?.on('data', (b: Buffer) => process.stderr.write(`   ! ${b}`));
  return child;
}

/** Speak the PeerJS handshake directly: no WebRTC needed to prove the broker
 *  registers a peer, derives its room, and drops it on disconnect. */
async function peerCheck(): Promise<void> {
  const id = 'skylens-demo-board-7';
  const url = `ws://localhost:8090/peerjs/peerjs?key=peerjs&id=${id}&token=tok123&version=1.5.5`;
  const ws = new WebSocket(url);
  await new Promise<void>((resolve) => {
    ws.on('message', (d: Buffer) => {
      console.log('   시그널링 응답:', d.toString());
      resolve();
    });
    ws.on('error', (e: Error) => {
      console.log('   시그널링 오류:', e.message);
      resolve();
    });
    setTimeout(resolve, 3000);
  });
  console.log('   등록 중 health.peer =', JSON.stringify((await health()).peer));
  ws.close();
  await sleep(800);
  console.log('   해제 후 health.peer =', JSON.stringify((await health()).peer));
}

async function main(): Promise<void> {
  let browser: Browser | null = null;
  let core: ChildProcess | null = null;
  try {
    browser = await chromium.launch({
      args: [
        '--use-gl=angle',
        '--use-angle=swiftshader',
        '--ignore-gpu-blocklist',
        '--enable-unsafe-swiftshader',
      ],
    });
    const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('console', (m) => {
      if (m.type() === 'error') errors.push(m.text());
    });

    // ---- 1. core DOWN --------------------------------------------------
    console.log('\n=== 1. 코어 정지 상태에서 현황판 기동 ===');
    console.log('   health.upstream =', JSON.stringify((await health()).upstream));
    await page.goto(BOARD, { waitUntil: 'load', timeout: 90_000 });
    await page.waitForFunction('window.skylens && window.skylens.splat', null, { timeout: 90_000 });
    await sleep(2500);
    let p = await probe(page);
    console.log(line(0, p));
    console.log(`   대기 배너: "${p.banner}"`);
    console.log(`   상태 문구: "${p.detail}"`);
    console.log(
      '   health.boards =',
      JSON.stringify((await health()).boards),
    );

    // ---- 2. core UP ----------------------------------------------------
    console.log('\n=== 2. 코어 기동 · 딜레이 패턴 스트림 ===');
    core = startCore();
    const t0 = Date.now();
    const seen: string[] = [];
    for (let i = 0; i < 55; i++) {
      await sleep(1000);
      p = await probe(page);
      const t = Math.round((Date.now() - t0) / 1000);
      const stamp = line(t, p);
      // Only print when something actually changed, so the timeline reads.
      const key = stamp.replace(/^t\+\s*\d+s/, '');
      if (!seen.includes(key)) {
        seen.push(key);
        console.log(stamp);
      }
      if (p.mission) {
        const m = `mission:${p.mission}`;
        if (!seen.includes(m)) {
          seen.push(m);
          console.log(`      임무 상태: ${p.mission}`);
        }
      }
    }

    console.log('\n   최종 사다리:');
    for (const s of p.segments) {
      console.log(`     구간 ${s.index + 1}: 수준 ${s.level}/${s.levels} · ${s.label}`);
    }
    console.log('   마커:', JSON.stringify(p.markers));
    console.log('   health.relayed =', JSON.stringify((await health()).relayed));

    // ---- 3. core DOWN again -------------------------------------------
    console.log('\n=== 3. 코어 중단 · 대기 상태 복귀 ===');
    stopCore(core);
    core = null;
    await sleep(6000);
    p = await probe(page);
    console.log(line(99, p));
    console.log(`   대기 배너: "${p.banner}"`);
    console.log(`   상태 문구: "${p.detail}"`);
    console.log(`   기하는 유지되는가: scenes=[${p.scenes.join(' ')}]`);
    console.log('   health.upstream =', JSON.stringify((await health()).upstream));

    // ---- 4. reload while core is down (replay cache) --------------------
    console.log('\n=== 4. 코어 정지 중 새로고침 (중계 캐시 재생) ===');
    await page.reload({ waitUntil: 'load', timeout: 90_000 });
    await page.waitForFunction('window.skylens && window.skylens.splat', null, { timeout: 90_000 });
    await sleep(9000);
    p = await probe(page);
    console.log(line(99, p));
    console.log('   health.relayed =', JSON.stringify((await health()).relayed));

    // ---- 5. WebRTC signalling (prepared, not live) ---------------------
    // Register a peer the way the PeerJS browser client does, to prove the
    // broker and its room bookkeeping actually work — this is the seam the
    // §2.3 board↔board redistribution will plug into.
    console.log('\n=== 5. PeerJS 시그널링 · 방/피어 집계 ===');
    await peerCheck();

    console.log('\n=== 페이지 오류 ===');
    console.log(errors.length ? errors.join('\n') : '   (없음)');
    // Best effort: a full splat scene on SwiftShader can take longer than the
    // default screenshot timeout, and a missing PNG must not fail the run.
    try {
      await page.screenshot({ path: 'C:/tmp/board-final.png', timeout: 120_000 });
      console.log('\n스크린샷: C:/tmp/board-final.png');
    } catch {
      console.log('\n스크린샷 생략 (렌더 타임아웃)');
    }
  } finally {
    if (core) stopCore(core);
    await browser?.close();
  }
}

void main();
