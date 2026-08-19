// Is everything on the situation board in the same place?
//
//   npm run demo
//   node src/test/client/boardSpaceCheck.mjs
//
// The board draws three things that all claim to describe one patch of ground:
// the aircraft, the reconstructed chunks, and the detections. They arrive by
// different routes — telemetry, the model's alignment, the model's detector —
// so they can disagree without anything erroring, and then the operator sees a
// flight in one corner and a reconstruction in the other.
//
// Everything is measured against the assigned track, in metres. The track is
// the one thing on the board that does not move, and the reconstruction
// legitimately TRAILS the aircraft by a segment or two — measuring chunks
// against the current drone position would fail that healthy lag and pass a
// chunk that happened to sit near the formation by accident.

import { chromium } from '@playwright/test';

const BOARD = process.env.SKYLENS_BOARD ?? 'http://localhost:8090/res/static/status.html';
const WAIT_MS = Number(process.env.SKYLENS_BOARD_WAIT ?? 90_000);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));

await page.goto(BOARD, { waitUntil: 'domcontentloaded', timeout: 120_000 });
await page.waitForFunction(() => window.skylens?.role === 'status', undefined, { timeout: 120_000 });
await page.bringToFront();

// Wait for a few segments: chunks only arrive once the formation has closed a
// segment and the model has returned it, and one chunk says nothing about
// whether the reconstruction follows the track.
await page
  .waitForFunction(() => (window.skylens.splat?.chunks ?? 0) >= 3, undefined, { timeout: WAIT_MS })
  .catch(() => console.log('[board] fewer than 3 chunks within the wait — reporting what there is'));
await page.waitForTimeout(2000);

const probe = await page.evaluate(async () => {
  const { sceneToEnu } = await import('/src/shared/geo.ts');
  const { state } = await import('/src/shared/viewer/store.ts');

  const drones = state.drones.map((d) => {
    const enu = sceneToEnu([d.pos.x, d.pos.y, d.pos.z]);
    return { station: d.station ?? d.id, e: Math.round(enu.e), n: Math.round(enu.n), u: Math.round(enu.u) };
  });
  const detections = state.detections.map((d) => {
    const enu = sceneToEnu(d.pos);
    return { kind: d.kind, e: Math.round(enu.e), n: Math.round(enu.n) };
  });
  const chunks = (window.skylens.splat?.loadedChunks?.() ?? []).map((c) => {
    const enu = sceneToEnu(c.center);
    return { segment: c.segment, level: c.level, e: Math.round(enu.e), n: Math.round(enu.n) };
  });
  const route = (state.route ?? []).map((p) => {
    const enu = sceneToEnu(p);
    return { e: Math.round(enu.e), n: Math.round(enu.n) };
  });
  return { drones, detections, chunks, route };
});

/** Distance from a point to the assigned track, in metres. */
const toRoute = (p) => {
  const line = probe.route;
  if (line.length < 2) return null;
  let best = Infinity;
  for (let i = 0; i + 1 < line.length; i++) {
    const ax = line[i].e;
    const ay = line[i].n;
    const bx = line[i + 1].e;
    const by = line[i + 1].n;
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const t = len2 > 0 ? Math.max(0, Math.min(1, ((p.e - ax) * dx + (p.n - ay) * dy) / len2)) : 0;
    best = Math.min(best, Math.hypot(p.e - (ax + dx * t), p.n - (ay + dy * t)));
  }
  return Math.round(best);
};
const worst = (pts) => (pts.length ? Math.max(...pts.map((p) => toRoute(p) ?? Infinity)) : null);

console.log('positions in metres from the board anchor (east, north):');
console.log('  route      :', JSON.stringify(probe.route));
console.log('  drones     :', JSON.stringify(probe.drones));
console.log('  chunks     :', JSON.stringify(probe.chunks));
console.log('  detections :', JSON.stringify(probe.detections));

const droneOff = worst(probe.drones);
const chunkOff = worst(probe.chunks);
const detOff = worst(probe.detections);
console.log('');
console.log('furthest each layer strays from the assigned track:');
console.log(`  drones     : ${droneOff ?? 'n/a'} m`);
console.log(`  chunks     : ${chunkOff ?? 'n/a'} m`);
console.log(`  detections : ${detOff ?? 'n/a'} m`);

console.log('');
console.log('===== RESULT =====');
const checks = [
  ['the board knows the assigned track', probe.route.length >= 2],
  ['the board has aircraft', probe.drones.length > 0],
  ['the board has a reconstruction', probe.chunks.length > 0],
  ['the board has detections', probe.detections.length > 0],
  // The wingmen hold station ~18 m off the track; chunks are anchored to the
  // middle of their own stretch of it; detections are a few tens of metres to
  // the side of the camera. Anything beyond that is a different piece of ground.
  [`the aircraft fly the track (${droneOff ?? '?'} m)`, droneOff != null && droneOff <= 40],
  [`the reconstruction sits on the track (${chunkOff ?? '?'} m)`, chunkOff != null && chunkOff <= 40],
  [`the detections sit along the track (${detOff ?? '?'} m)`, detOff != null && detOff <= 60],
  ['no page errors', errors.length === 0],
];
for (const [name, ok] of checks) console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`);
if (errors.length) console.log(errors.join('\n'));

await browser.close();
process.exit(checks.every(([, ok]) => ok) ? 0 : 1);
