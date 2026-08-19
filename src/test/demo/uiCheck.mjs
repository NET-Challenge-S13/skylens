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

const fleet = await tower.evaluate(() => ({
  drones: window.skylens.state.drones.map((d) => d.id),
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

await tower.screenshot({ path: `${OUT}/ui-tower.png` });
await board.screenshot({ path: `${OUT}/ui-board.png` });

console.log('');
console.log('===== RESULT =====');
const checks = [
  ['formation is 3 aircraft', fleet.drones.length === 3],
  ['MAIN CAM is playing footage', !!fleet.video && !fleet.video.hidden && fleet.video.time > 0],
  ['minimap clear of the detection card', !boardState.overlaps],
  ['board received geometry', boardState.chunks > 0],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));
console.log(`screenshots: ${OUT}`);

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
