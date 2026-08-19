// Route planner: where the map opens, and whether it can be dragged.
//
//   npm run demo                                  (one shell — needs the core)
//   node src/test/control/plannerCheck.mjs [--headed]
//
// Everything here is read the way an operator would see it: waypoints are
// placed by clicking the canvas and their coordinates read back off the list,
// so a passing check means the map really is looking where it claims.

import { chromium } from '@playwright/test';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';
const CORE = process.env.SKYLENS_CORE ?? 'http://localhost:8080';
const headed = process.argv.includes('--headed');

const log = (...a) => console.log('[planner]', ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const site = await (await fetch(`${CORE}/site`)).json();
log('core site:', JSON.stringify(site));

const browser = await chromium.launch({ headless: !headed });
// A fresh context every run: "no cache" is the state under test, and a leftover
// stored centre would quietly make the first assertion vacuous.
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.bringToFront();
await sleep(4000);

await page.evaluate(() => window.skylens.routeModal.open());
await sleep(1500);

const canvas = page.locator('.route-modal__canvas');
const box = await canvas.boundingBox();
const mid = { x: box.x + box.width / 2, y: box.y + box.height / 2 };

/** Place a waypoint at a screen point and read back what the planner recorded. */
async function placeAndRead(at) {
  await page.mouse.click(at.x, at.y);
  await sleep(250);
  const items = await page.locator('.route-modal__item span').allTextContents();
  const last = items[items.length - 1] ?? '';
  const m = /(-?\d+\.\d+),\s*(-?\d+\.\d+)/.exec(last);
  return m ? { lat: Number(m[1]), lon: Number(m[2]), count: items.length } : null;
}

// 1. With no stored centre, the map should be looking at the core's site.
const atCentre = await placeAndRead(mid);
log('waypoint at map centre:', JSON.stringify(atCentre));
const nearSite =
  atCentre !== null &&
  Math.abs(atCentre.lat - site.gps.lat) < 0.01 &&
  Math.abs(atCentre.lon - site.gps.lon) < 0.01;

// 2. Drag the map. The same screen point must then be different ground, and the
//    drag itself must not leave a waypoint behind.
const before = atCentre?.count ?? 0;
await page.mouse.move(mid.x, mid.y);
await page.mouse.down();
for (let i = 1; i <= 8; i++) {
  await page.mouse.move(mid.x - i * 12, mid.y - i * 6);
  await sleep(30);
}
await page.mouse.up();
await sleep(600);

const afterDragCount = (await page.locator('.route-modal__item span').allTextContents()).length;
const draggedSilently = afterDragCount === before;

const afterDrag = await placeAndRead(mid);
log('waypoint at the same point after dragging:', JSON.stringify(afterDrag));
const mapMoved =
  atCentre !== null &&
  afterDrag !== null &&
  Math.hypot(afterDrag.lat - atCentre.lat, afterDrag.lon - atCentre.lon) > 1e-4;

// 3. The pan is remembered for next time.
const stored = await page.evaluate(() => localStorage.getItem('skylens.control.mapCenter.v1'));
log('stored centre:', stored);

console.log('');
console.log('===== RESULT =====');
const checks = [
  ['opens on the core site when nothing is cached', nearSite],
  ['dragging moves the map', mapMoved],
  ['dragging does not drop a waypoint', draggedSilently],
  ['the pan is remembered', stored !== null && /lat/.test(stored)],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
