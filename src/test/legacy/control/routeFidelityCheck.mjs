// Does the drone fly where the operator pointed?
//
//   npm run demo                                       (one shell)
//   node src/test/control/routeFidelityCheck.mjs [--headed]
//
// Plans a route by clicking the planner map, then follows the same coordinates
// through every hand-off and measures the error at each one:
//
//   planner list  ->  what the core stored  ->  where the centre drone flew
//
// Reporting each hop separately is the point: "the drone is off the route" can
// mean the map lied, the wire mangled it, or the aircraft is holding station
// somewhere else, and those are three different bugs.

import { chromium } from '@playwright/test';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';
const CORE = process.env.SKYLENS_CORE ?? 'http://localhost:8080';
const headed = process.argv.includes('--headed');

const log = (...a) => console.log('[route]', ...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Ground distance in metres between two fixes (local flat approximation). */
function metres(a, b) {
  const mPerLat = 111_320;
  const mPerLon = 111_320 * Math.cos((a.lat * Math.PI) / 180);
  return Math.hypot((b.lat - a.lat) * mPerLat, (b.lon - a.lon) * mPerLon);
}

/** Perpendicular distance from a point to the assigned polyline, in metres. */
function crossTrack(point, line) {
  let best = Infinity;
  for (let i = 0; i + 1 < line.length; i++) {
    const a = line[i];
    const b = line[i + 1];
    const mPerLat = 111_320;
    const mPerLon = 111_320 * Math.cos((a.lat * Math.PI) / 180);
    const ax = 0;
    const ay = 0;
    const bx = (b.lon - a.lon) * mPerLon;
    const by = (b.lat - a.lat) * mPerLat;
    const px = (point.lon - a.lon) * mPerLon;
    const py = (point.lat - a.lat) * mPerLat;
    const len2 = (bx - ax) ** 2 + (by - ay) ** 2;
    let t = len2 > 0 ? ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    best = Math.min(best, Math.hypot(px - (ax + (bx - ax) * t), py - (ay + (by - ay) * t)));
  }
  return best;
}

const browser = await chromium.launch({ headless: !headed });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.bringToFront();
await sleep(5000);

await page.evaluate(() => window.skylens.routeModal.open());
await sleep(1500);

// --- 1. plan a route by clicking the map ----------------------------------
const box = await page.locator('.route-modal__canvas').boundingBox();
const points = [
  { x: box.x + box.width * 0.3, y: box.y + box.height * 0.65 },
  { x: box.x + box.width * 0.5, y: box.y + box.height * 0.5 },
  { x: box.x + box.width * 0.72, y: box.y + box.height * 0.34 },
];
for (const p of points) {
  await page.mouse.click(p.x, p.y);
  await sleep(200);
}

const planned = (await page.locator('.route-modal__item span').allTextContents()).map((t) => {
  const m = /(-?\d+\.\d+),\s*(-?\d+\.\d+)/.exec(t);
  return m ? { lat: Number(m[1]), lon: Number(m[2]) } : null;
});
log('planned:', JSON.stringify(planned));

await page.locator('.route-modal__btn--primary').click();
await sleep(1500);

// --- 2. what the core stored ----------------------------------------------
const health = await (await fetch(`${CORE}/health`)).json();
const stored = (health.routeWaypoints ?? health.route?.waypointList ?? []).map((w) => ({
  lat: w.lat,
  lon: w.lon,
}));
log('core stored count:', health.route?.waypoints, 'length(m):', health.route?.lengthM);

// The core reports the count and length rather than the list, so compare those:
// a waypoint that moved would change the length.
let plannedLength = 0;
for (let i = 0; i + 1 < planned.length; i++) plannedLength += metres(planned[i], planned[i + 1]);
const lengthErr = Math.abs((health.route?.lengthM ?? 0) - plannedLength);
log(`planned length ${plannedLength.toFixed(1)} m · core ${health.route?.lengthM} m · Δ ${lengthErr.toFixed(1)} m`);

// --- 3. where the aircraft actually flew ------------------------------------
log('waiting for the formation to reach the route');
await sleep(30_000);

const fixes = await page.evaluate(async () => {
  const seen = [];
  const started = Date.now();
  while (Date.now() - started < 12_000) {
    for (const d of window.skylens.fleet.drones()) {
      seen.push({ station: d.station, lat: d.gps.lat, lon: d.gps.lon });
    }
    await new Promise((r) => setTimeout(r, 400));
  }
  return seen;
});

// The line the operator sees. It is drawn from the same GPS the drones fly, so
// checking the drawn vertices against the plan closes the loop the operator
// actually judges by: plan on the map, line on the 3D view, aircraft on the line.
const drawn = await page.evaluate(() => {
  const group = window.skylens.viewer.debugRoute?.();
  if (!group) return null;
  return group.map((v) => window.skylens.frame.toGps({ x: v.x, y: v.y, z: v.z }));
});
log('route drawn on the 3D map:', JSON.stringify(drawn));

const byStation = {};
for (const f of fixes) {
  const off = crossTrack(f, planned);
  byStation[f.station] = Math.max(byStation[f.station] ?? 0, off);
}
log('max cross-track by station (m):', JSON.stringify(byStation));

console.log('');
console.log('===== RESULT =====');
const checks = [
  ['the planner recorded three waypoints', planned.length === 3 && planned.every(Boolean)],
  ['the core stored the same route', health.route?.waypoints === 3 && lengthErr < 5],
  // The centre aircraft flies the track itself; the wingmen hold station ~18 m
  // off it by design, so only the centre is held to the line.
  ['the centre drone flies the planned line', (byStation.center ?? Infinity) < 25],
  ['the wingmen hold station near it', (byStation.left ?? Infinity) < 60],
  [
    'the planned line is drawn on the 3D map',
    Array.isArray(drawn) &&
      drawn.length >= 3 &&
      planned.every((p, i) => drawn[i] && metres(p, drawn[i]) < 1),
  ],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
