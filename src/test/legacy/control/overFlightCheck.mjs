// Does the aircraft actually cross the building the route was drawn on?
//
//   npm run demo
//   node src/test/control/overFlightCheck.mjs
//
// Every link in the chain measures clean on its own — the planner's pixels, the
// wire, the scene frame, the footprints. This asks the operator's question
// instead: a route drawn along 충남대 제5공학관 must put the aircraft over that
// building's roof. It finds the footprint under the waypoints, flies the route,
// and reports how close the ground track comes to that polygon.

import { chromium } from '@playwright/test';

const TOWER = process.env.SKYLENS_TOWER ?? 'http://localhost:8080/res/static/control.html';

// The operator's own waypoints, from the report.
const WAYPOINTS = [
  { lat: 36.36644, lon: 127.34523 },
  { lat: 36.36649, lon: 127.34478 },
  { lat: 36.36652, lon: 127.34446 },
  { lat: 36.36656, lon: 127.34405 },
  { lat: 36.36662, lon: 127.34369 },
];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
await page.goto(TOWER, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'control', undefined, { timeout: 120_000 });
await page.bringToFront();
await page.waitForTimeout(20_000);

// --- which buildings lie under the drawn route -----------------------------
const under = await page.evaluate(async (wps) => {
  const mLat = 111_320;
  const mLon = 111_320 * Math.cos((wps[0].lat * Math.PI) / 180);
  const lats = wps.map((w) => w.lat);
  const lons = wps.map((w) => w.lon);
  const pad = 0.0006;
  const url =
    `/vworld/wfs?SERVICE=WFS&VERSION=1.1.0&REQUEST=GetFeature&TYPENAME=lt_c_bldginfo` +
    `&SRSNAME=EPSG:4326&BBOX=${Math.min(...lats) - pad},${Math.min(...lons) - pad},` +
    `${Math.max(...lats) + pad},${Math.max(...lons) + pad},EPSG:4326` +
    `&maxFeatures=300&OUTPUT=application/json`;
  const res = await fetch(url);
  const feats = res.ok ? ((await res.json()).features ?? []) : [];

  /** Shortest distance from a fix to a polygon's edges, in metres (0 inside). */
  const distTo = (ring, p) => {
    let inside = false;
    let best = Infinity;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [x1, y1] = ring[i];
      const [x2, y2] = ring[j];
      if (y1 > p.lat !== y2 > p.lat && p.lon < ((x2 - x1) * (p.lat - y1)) / (y2 - y1) + x1) {
        inside = !inside;
      }
      const ax = (x1 - p.lon) * mLon;
      const ay = (y1 - p.lat) * mLat;
      const bx = (x2 - p.lon) * mLon;
      const by = (y2 - p.lat) * mLat;
      const dx = bx - ax;
      const dy = by - ay;
      const len2 = dx * dx + dy * dy;
      const t = len2 > 0 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / len2)) : 0;
      best = Math.min(best, Math.hypot(ax + dx * t, ay + dy * t));
    }
    return inside ? 0 : best;
  };

  const out = [];
  for (const f of feats) {
    const polys =
      f.geometry?.type === 'MultiPolygon' ? f.geometry.coordinates : [f.geometry?.coordinates];
    for (const poly of polys ?? []) {
      const ring = poly?.[0];
      if (!ring || ring.length < 3) continue;
      const perWaypoint = wps.map((w) => Math.round(distTo(ring, w)));
      out.push({
        id: f.id,
        name: f.properties?.dong_nm ?? null,
        floors: f.properties?.grnd_flr ?? null,
        height: f.properties?.height ?? null,
        area: f.properties?.archarea ?? null,
        ring,
        nearestWaypointM: Math.min(...perWaypoint),
        waypointsOver: perWaypoint.filter((d) => d === 0).length,
      });
    }
  }
  out.sort((a, b) => a.nearestWaypointM - b.nearestWaypointM);
  return out.slice(0, 5).map(({ ring, ...rest }) => ({ ...rest, corners: ring.length }));
}, WAYPOINTS);

console.log('buildings under the drawn route (nearest first):');
for (const b of under) {
  console.log(
    `  ${(b.name ?? '(이름 없음)').padEnd(22)} ${String(b.floors ?? '?').padStart(2)}층 · ` +
      `${String(b.area ?? '?').padStart(7)} m2 · 웨이포인트까지 ${b.nearestWaypointM} m · ` +
      `건물 위 웨이포인트 ${b.waypointsOver}/${WAYPOINTS.length}`,
  );
}

// --- fly it and follow the centre aircraft ---------------------------------
await page.evaluate(async (wps) => {
  const frame = window.skylens.frame;
  const waypoints = wps.map((w) => ({
    ...w,
    alt: Math.round(frame.groundAltAt({ ...w, alt: 0 }) + 60),
  }));
  const leader = window.skylens.fleet.drones().find((d) => d.station === 'center');
  window.skylens.core.send({ kind: 'assign-route', droneId: leader?.id ?? 1, waypoints, loop: true });
}, WAYPOINTS);

console.log('');
console.log('route assigned — following the centre aircraft for a full leg');
await page.waitForTimeout(30_000);

const track = await page.evaluate(async () => {
  const seen = [];
  const started = Date.now();
  while (Date.now() - started < 45_000) {
    const d = window.skylens.fleet.drones().find((x) => x.station === 'center');
    if (d) seen.push({ lat: d.gps.lat, lon: d.gps.lon });
    await new Promise((r) => setTimeout(r, 300));
  }
  return seen;
});

// How close the track came to each waypoint — the plainest reading of "did it
// go where I pointed".
const mLat = 111_320;
const mLon = 111_320 * Math.cos((WAYPOINTS[0].lat * Math.PI) / 180);
const closest = WAYPOINTS.map((w) => {
  let best = Infinity;
  for (const p of track) {
    best = Math.min(best, Math.hypot((p.lon - w.lon) * mLon, (p.lat - w.lat) * mLat));
  }
  return Math.round(best * 10) / 10;
});

console.log(`track samples: ${track.length}`);
console.log('closest approach to each waypoint (m):', JSON.stringify(closest));

console.log('');
console.log('===== RESULT =====');
const checks = [
  ['the route was drawn over a real building', under.length > 0 && under[0].nearestWaypointM <= 5],
  ['the aircraft reached every waypoint (<15 m)', closest.every((d) => d < 15)],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
