// Visual check of the running demo, for the things only a rendered frame can
// answer: is the formation actually three aircraft, does MAIN CAM play, does the
// minimap sit clear of the detection card, and is the type one family?
//
//   npm run demo                                (one shell)
//   node src/test/demo/uiCheck.mjs [--headed]   (another)
//
// It drives the scenario the way an operator would — assign a route in the
// control tower, then watch both screens — and writes screenshots for review.

import { chromium } from '@playwright/test';
import { mkdirSync } from 'node:fs';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';
const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const OUT = process.env.SKYLENS_SHOTS ?? 'C:/tmp/skylens-shots';
const headed = process.argv.includes('--headed');

mkdirSync(OUT, { recursive: true });
const log = (...a) => console.log('[ui]', ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch({ headless: !headed });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });

const tower = await ctx.newPage();
const errors = [];
tower.on('pageerror', (e) => errors.push(`tower pageerror: ${e.message}`));

await tower.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await tower.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
log('tower booted');
await sleep(6000);

// Assign the demo route through the real modal path.
const assigned = await tower.evaluate(async () => {
  const { DEMO_ROUTE } = await import('/src/skylens_drone/core/config.ts');
  return window.skylens.core.send({
    kind: 'assign-route',
    droneId: 1,
    waypoints: DEMO_ROUTE,
    loop: true,
  });
});
log('route assigned:', assigned);

const board = await ctx.newPage();
board.on('pageerror', (e) => errors.push(`board pageerror: ${e.message}`));
await board.goto(BOARD, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await board.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
log('board booted');

// The drone reports in ~10 s after assignment, then flies. Give the formation
// time to spread and the first slices time to reach the core.
await sleep(30_000);

// Switch the camera to each station in turn and record what the panel shows.
// The tower has to be the FRONT tab first: its update loop runs on rAF, which a
// background tab barely gets, so a switch made here would not be applied before
// the assertion reads it back.
await tower.bringToFront();
await sleep(500);
const cams = [];
for (const station of ['left', 'center', 'right']) {
  // Select the way the operator does — through the fleet list — so the event
  // path the UI actually depends on is what gets exercised.
  const picked = await tower.evaluate(async (want) => {
    const { emit, state } = await import('/src/shared/viewer/store.ts');
    const d = window.skylens.fleet.drones().find((x) => x.station === want);
    if (!d) return null;
    state.activeDroneId = d.id;
    emit({ type: 'active-drone', id: d.id });
    return d.id;
  }, station);
  if (picked === null) {
    cams.push({ station, ok: false, why: 'no drone at this station' });
    continue;
  }
  await sleep(1500);
  const shown = await tower.evaluate(() => ({
    active: window.skylens.state.activeDroneId,
    label: document.querySelector('.video-panel__label')?.textContent ?? '',
    src: (document.querySelector('.video-panel__video')?.getAttribute('src') ?? '').split('/').pop(),
    playing: !document.querySelector('.video-panel__video')?.classList.contains('is-hidden'),
    overlay: document.querySelector('.video-panel__overlay')?.textContent ?? '',
  }));
  cams.push({ station, ...shown, ok: shown.label === `${station.toUpperCase()} CAM` && shown.playing });
  log('cam', station, JSON.stringify(shown));
}

const fleet = await tower.evaluate(() => ({
  drones: window.skylens.state.drones.map((d) => d.id),
  stations: window.skylens.fleet.drones().map((d) => d.station).sort(),
  video: (() => {
    const v = document.querySelector('.video-panel__video');
    return v
      ? {
          hidden: v.classList.contains('is-hidden'),
          src: (v.getAttribute('src') ?? '').split('/').pop(),
          time: Number(v.currentTime.toFixed(2)),
          w: v.videoWidth,
          h: v.videoHeight,
        }
      : null;
  })(),
  overlay: document.querySelector('.video-panel__overlay')?.textContent ?? '',
  label: (() => {
    const el = document.querySelector('.pane__label');
    const s = el ? getComputedStyle(el) : null;
    return s ? { font: s.fontFamily, tracking: s.letterSpacing, weight: s.fontWeight } : null;
  })(),
  panelFont: (() => {
    const el = document.querySelector('.mission-panel__message, .mission-panel');
    const s = el ? getComputedStyle(el) : null;
    return s ? { font: s.fontFamily } : null;
  })(),
}));
log('tower:', JSON.stringify(fleet));

await board.bringToFront();
await sleep(500);
const boardState = await board.evaluate(() => {
  const mm = document.getElementById('minimap')?.getBoundingClientRect();
  const card = document.querySelector('.detect-card')?.getBoundingClientRect();
  const overlaps =
    mm && card
      ? !(mm.right < card.left || mm.left > card.right || mm.bottom < card.top || mm.top > card.bottom)
      : false;
  return {
    drones: window.skylens.state.drones.length,
    detections: window.skylens.state.detections.length,
    chunks: window.skylens.splat?.chunks ?? 0,
    minimap: mm ? { top: Math.round(mm.top), right: Math.round(mm.right) } : null,
    detectCard: card ? { top: Math.round(card.top), right: Math.round(card.right) } : null,
    overlaps,
  };
});
log('board:', JSON.stringify(boardState));

await tower.bringToFront();
await sleep(800);
await tower.screenshot({ path: `${OUT}/ui-tower.png` });
await board.screenshot({ path: `${OUT}/ui-board.png` });

console.log('');
console.log('===== RESULT =====');
const checks = [
  ['formation is 3 aircraft', fleet.drones.length === 3],
  ['stations are left/center/right', fleet.stations.join(',') === 'center,left,right'],
  ['each station has its own camera', cams.length === 3 && cams.every((c) => c.ok)],
  [
    'the three cameras show different footage',
    new Set(cams.map((c) => c.src)).size === 3,
  ],
  ['the selected camera is playing', cams.every((c) => c.playing)],
  ['minimap clear of the detection card', !boardState.overlaps],
  ['board received geometry', boardState.chunks > 0],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));
console.log(`screenshots: ${OUT}`);

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
